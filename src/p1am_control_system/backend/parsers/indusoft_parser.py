import json
from pathlib import Path

try:
    from ..plant_model import TagDefinition
except (ImportError, ValueError):
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent))
    from plant_model import TagDefinition  # type: ignore[no-redef]


def parse_indusoft_tags(json_path: Path) -> list[TagDefinition]:
    """
    Parses an InduSoft tagl.json file and returns a list of TagDefinition DbC models.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Tag database not found at {json_path}")

    with open(json_path, encoding="utf-16") as f:
        raw_data = json.load(f)

    tags = []
    for t in raw_data.get("tags", []):
        tag_type = t.get("type", "Real")
        if tag_type not in ("Real", "Boolean", "Integer", "String"):
            tag_type = "Real"  # fallback

        tags.append(
            TagDefinition(
                name=t.get("name", "Unknown"),
                tag_type=tag_type,
                description=t.get("description", ""),
                rw_mode=(
                    "Read/Write"
                    if t.get("external_availability") == "Enabled"
                    else "Read-only"
                ),
            )
        )
    return tags
