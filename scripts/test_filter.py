from pathlib import Path

from data_miner.config import SIGLIP2_MODELS, FilterConfig
from data_miner.modules.frame_filter import FrameFilter

# threshold=0.35,
# negative_threshold=0.45,
# margin_threshold=0.12,
# zoom_threshold=0.4,
# zoom_margin_threshold=0.1,

negative_prompts_zoom = [
    # --- NEW: The "Zoom Trap" ---
    "full door not visible",
    # "more than 50% of door cropped out",
    # "large portion of door cropped out",
    "extreme close up of a glass door",
    "macro shot of a door handle",
    "zoomed in detail of door hardware",
    "a close up texture of glass",
    "cropped view of a metal frame",
    "partial view of a door edge",
    "a photo showing only the door handle",
    "abstract architectural detail",
    "view of a hinge or lock only",
]

positive_prompts_full_view = [
    # Emphasize "Full", "Entire", "Wide"
    "a full view of a glass entrance door",
    "entire glass door clearly visible from top to bottom",
    "a wide shot of a glass storefront entrance",
    "commercial double glass doors showing the floor", # "Showing floor" is a great heuristic for full view
    "a complete sliding glass door",
    "framed glass door fully visible"
]

zoom_prompts = [
    "extreme close up", "macro shot", "zoomed in detail", 
    "texture only", "cropped view"
]

positive_prompts = [
    "a glass door",
    "a french door",
    "a patio door",
    "a photo of a glass door",
    "a glass entrance door",
    "a glass entrance door with handles",
    "commercial double glass doors",
    "a sliding glass door entrance",
    "a storefront entrance with a glass door",
    "a building entrance with a glass door",
    "framed glass door clearly visible",
    "a push bar on a glass door",
]
negative_prompts = [
    "a glass wall",
    "a fixed glass window",
    "a large display window",
    "a glass curtain wall",
    "a mirror",
    "a reflective surface",
    "a shower door",
    "a glass partition",
    "a skylight",
    "a glass ceiling",
    "a split screen video",
    "text overlay on screen",
    "a video game screenshot",
    "blurry out of focus image",
    "dark grainy night shot",
]

# Category 1: want to KEEP (full/usable views for detection training)
full_view_positive_prompts = [
    "a full glass door",
    "a glass door fully visible",
    "the entire glass door is in the frame",
    "a wide shot of a glass door",
    "a wide shot of an entrance with a glass door",
    "a storefront entrance with a glass door",
    "a building entrance with a glass door",
    "commercial double glass doors fully visible",
    "a sliding glass door with the full door visible",
    "a french door fully visible",
    "a patio door fully visible",
    "a framed glass door fully visible",
    "a glass entrance door fully visible",
    "a glass door with surrounding context visible",
]

# Category 2: close-up / zoomed / cropped (want to REJECT)
closeup_negative_prompts = [
    "a close-up of a glass door",
    "an extreme close-up of a glass door",
    "a zoomed-in photo of a glass door",
    "a tightly cropped glass door",
    "only part of a glass door is visible",
    "a partial view of a glass door",
    "a close-up of a door handle",
    "a close-up of a glass door handle",
    "a close-up of a push bar",
    "a close-up of a door frame",
    "a close-up of a door hinge",
]

# Category 3: not-a-door / low-quality / irrelevant (want to REJECT)
non_door_or_bad_negative_prompts = [
    "a glass wall",
    "a fixed glass window",
    "a large display window",
    "a glass curtain wall",
    "a mirror",
    "a reflective surface",
    "a shower door",
    "a glass partition",
    "a skylight",
    "a glass ceiling",
    "a split screen video",
    "text overlay on screen",
    "a video game screenshot",
    "blurry out of focus image",
    "dark grainy night shot",
]
input_dir = Path("/mnt/data_2/pavan/project_helpers/data_miner/output/projects/direct_doors_v1/frames_dedup")

filter_conf = FilterConfig(
    output_dir=input_dir.parent,
    device="cuda:7",
    model_id=SIGLIP2_MODELS["siglip2-giant"],
    batch_size=8,
    positive_prompts=full_view_positive_prompts,
    threshold=0.3,
    negative_prompts=non_door_or_bad_negative_prompts,
    negative_threshold=0.3,
    margin_threshold=0.1,
    zoom_prompts=closeup_negative_prompts,
    zoom_threshold=0.3,
    zoom_margin_threshold=0.1,
)
frame_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))

_filter = FrameFilter(config=filter_conf, device_map=filter_conf.device)

result = _filter.filter_frames(
    frame_paths=frame_paths,
    video_id="frames_dedup_filter_v3",
)
print(f"\n {result.passed_frames}/{result.total_frames} frames passed")
print("Completed")

