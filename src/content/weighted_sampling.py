import json
from pathlib import Path

def load_catmus_chars(*json_paths):
    chars = set()
    for path in json_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data.get('characters', []):
                c = entry.get('character', '')
                if c:
                    chars.update(c)  # update car certains chars sont multi-codepoints
        except FileNotFoundError:
            print(f"  Warning: CATMuS file not found: {path}")
    return chars

CATMUS_SPECIAL_CHARS = load_catmus_chars(
    Path(__file__).parent / "catmus-combining.json",
    Path(__file__).parent / "catmus-medieval.json",
    Path(__file__).parent / "catmus-superscript.json",
)
print(f"  Loaded {len(CATMUS_SPECIAL_CHARS)} special CATMuS characters")

def special_char_density(text):
    if not text:
        return 0.0
    count = sum(1 for c in text if c in CATMUS_SPECIAL_CHARS)
    return count / len(text)

