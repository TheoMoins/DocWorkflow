import re
import json
import unicodedata
from pathlib import Path
from lxml import etree as ET

# Load CATMuS characters from bundled JSON files
_DATA_DIR = Path(__file__).parent.parent.parent / "content" / "catmus"

def _build_catmus_whitelist() -> set:
    """Build the set of allowed characters from CATMuS JSON files + Latin standard."""
    allowed = set()

    # CATMuS special characters from JSON files
    for json_file in _DATA_DIR.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("characters", []):
            char = entry.get("character", "")
            for c in char:  # some entries are multi-codepoint
                allowed.add(c)

    # Latin standard: Basic Latin + Latin Extended A/B + Latin Extended Additional
    # Unicode blocks U+0020–U+024F + U+1E00–U+1EFF
    for codepoint in range(0x0020, 0x0250):
        c = chr(codepoint)
        cat = unicodedata.category(c)
        if cat[0] in ('L', 'N', 'P', 'Z', 'M'):
            allowed.add(c)
    for codepoint in range(0x1E00, 0x1F00):
        c = chr(codepoint)
        if unicodedata.category(c)[0] in ('L', 'M'):
            allowed.add(c)

    # Always allow whitespace and line separators
    allowed.update({' ', '\t', '\n', '\r'})

    return allowed


CATMUS_WHITELIST: set = _build_catmus_whitelist()


def clean_htr_text(text: str) -> str:
    """
    Apply CATMuS post-processing rules to a single string:
    1. Remove '[...]' entirely
    2. Strip brackets from '[other content]', keeping the content
    3. Remove '<tag>' and their content entirely
    4. Remove characters not in the CATMuS whitelist
    """
    # 1. Remove '[...]' entirely (literal ellipsis or dots)
    text = re.sub(r'\[\.{2,}\]', '', text)
    text = re.sub(r'\[…\]', '', text)

    # 2. Remove angle-bracket tags and their content
    text = re.sub(r'<[^>]*>', '', text)

    # 3. Strip brackets but keep content
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)

    # 4. Filter to whitelist
    text = ''.join(c for c in text if c in CATMUS_WHITELIST)

    return text


def clean_alto_file(alto_path: str) -> None:
    """
    Apply CATMuS post-processing in-place on all String CONTENT attributes
    in an ALTO XML file.
    """
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

    for string_elem in root.findall('.//alto:String', ns):
        content = string_elem.get('CONTENT', '')
        if content:
            string_elem.set('CONTENT', clean_htr_text(content))

    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")