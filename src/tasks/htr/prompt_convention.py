
import json
from pathlib import Path


CONVENTION_FRAGMENTS = {
    "abbreviations": {
        "keep": "Keep abbreviations as written, do not expand them.",
        "resolve": "Resolve and expand all abbreviations.",
    },
    "word_segmentation": {
        "original": "Preserve the original word segmentation.",
        "modern": "Modernize word segmentation (split or join words following modern usage).",
    },
    "uv_ij": {
        "original": "Keep original u/v and i/j distinctions as written.",
        "modern": "Modernize u/v and i/j usage.",
        "ui_only": "Use only u and i, never v and j, regardless of the original or modern usage.",
    },
    "allographs": {
        True: "Preserve allographic variants (e.g. long s 'ſ', special letter forms).",
        False: "Do not record allographic variants, use standard letter forms.",
    },
}


def build_conventions_block(conventions: dict) -> str:
    """Turn a conventions dict into a prompt fragment."""
    parts = []
    for key, mapping in CONVENTION_FRAGMENTS.items():
        value = conventions.get(key)
        if value is not None and value in mapping:
            parts.append(mapping[value])
    return "\n".join(parts)

def load_conventions(data_path) -> dict:
    """Load conventions from metadata.json in a data directory."""
    meta_path = Path(data_path) / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("conventions", {})