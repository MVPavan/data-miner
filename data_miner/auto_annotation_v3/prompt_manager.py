from pathlib import Path
import hashlib
import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptTemplate:
    def __init__(self, data: dict):
        self.id = data["id"]
        self.version = data["version"]
        self.system = data["system"]
        self.variables = data.get("variables", [])
        self.model_params = data.get("model_params", {})
        self.stage = data.get("stage", "")
        self.group = data.get("group")
        self.changelog = data.get("changelog", "")

    def render(self, **kwargs) -> str:
        return self.system.format(**kwargs)

    def render_and_hash(self, **kwargs) -> tuple[str, str]:
        rendered = self.render(**kwargs)
        prompt_hash = hashlib.sha256(rendered.encode()).hexdigest()[:12]
        return rendered, prompt_hash


def load_prompt(prompt_id: str, version_dir: Path = None) -> PromptTemplate:
    """Load prompt with fallback to parent version."""
    if version_dir is None:
        version_dir = PROMPTS_DIR / "active"
    version_dir = version_dir.resolve()

    path = version_dir / f"{prompt_id}.yaml"
    if path.exists():
        return PromptTemplate(yaml.safe_load(path.read_text()))

    manifest = yaml.safe_load((version_dir / "manifest.yaml").read_text())
    parent = manifest.get("parent")
    if parent:
        return load_prompt(prompt_id, PROMPTS_DIR / parent)

    raise FileNotFoundError(
        f"Prompt '{prompt_id}' not found in {version_dir} or parents"
    )


def _collect_prompt_ids(version_dir: Path, seen: set[str]) -> set[str]:
    """Recursively collect all prompt IDs reachable from version_dir."""
    version_dir = version_dir.resolve()
    for f in version_dir.glob("*.yaml"):
        if f.name != "manifest.yaml":
            seen.add(f.stem)
    manifest_path = version_dir / "manifest.yaml"
    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text())
        parent = manifest.get("parent")
        if parent:
            _collect_prompt_ids(PROMPTS_DIR / parent, seen)
    return seen


def load_all_active_prompts() -> dict[str, PromptTemplate]:
    """Load all prompts from the active version (with inheritance fallback)."""
    active_dir = PROMPTS_DIR / "active"
    prompt_ids = _collect_prompt_ids(active_dir, set())
    return {pid: load_prompt(pid) for pid in sorted(prompt_ids)}


def compute_prompt_hash(prompts_dir: Path = None) -> str:
    """Hash all active prompt files for config comparison."""
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR
    active = (prompts_dir / "active").resolve()
    h = hashlib.sha256()
    for f in sorted(active.glob("*.yaml")):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]
