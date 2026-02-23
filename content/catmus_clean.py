#!/usr/bin/env python3
"""
Standalone CATMuS post-processing script.
Applies cleaning rules to ALTO XML prediction files in-place (or to a copy).

Usage:
    python catmus_clean.py <input_dir> [--output <output_dir>]

    If --output is omitted, files are modified in-place.
    The three CATMuS JSON files must be in the same directory as this script.
"""

import re
import json
import unicodedata
import argparse
import shutil
import glob
from pathlib import Path
from lxml import etree as ET


# ---------------------------------------------------------------------------
# Whitelist builder
# ---------------------------------------------------------------------------

def _build_catmus_whitelist(json_dir: Path) -> set:
    allowed = set()

    for json_file in json_dir.glob("catmus-*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("characters", []):
            for c in entry.get("character", ""):
                allowed.add(c)

    # Latin standard: Basic Latin + Latin Extended A/B (U+0020–U+024F)
    for cp in range(0x0020, 0x0250):
        c = chr(cp)
        if unicodedata.category(c)[0] in ('L', 'N', 'P', 'Z', 'M'):
            allowed.add(c)
    # Latin Extended Additional (U+1E00–U+1EFF)
    for cp in range(0x1E00, 0x1F00):
        c = chr(cp)
        if unicodedata.category(c)[0] in ('L', 'M'):
            allowed.add(c)

    allowed.update({' ', '\t', '\n', '\r'})
    return allowed


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_htr_text(text: str, whitelist: set) -> str:
    # 1. Remove '[...]' / '[…]' entirely
    text = re.sub(r'\[\.{2,}\]', '', text)
    text = re.sub(r'\[…\]', '', text)

    # 2. Remove '<tag>' and their content
    text = re.sub(r'<[^>]*>', '', text)

    # 3. Strip brackets, keep content
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)

    # 4. Filter to whitelist
    text = ''.join(c for c in text if c in whitelist)

    return text


# ---------------------------------------------------------------------------
# ALTO file processing
# ---------------------------------------------------------------------------

def clean_alto_file(alto_path: str, whitelist: set) -> int:
    """Returns number of String elements modified."""
    tree = ET.parse(alto_path)
    root = tree.getroot()
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}

    modified = 0
    for string_elem in root.findall('.//alto:String', ns):
        content = string_elem.get('CONTENT', '')
        if content:
            cleaned = clean_htr_text(content, whitelist)
            if cleaned != content:
                string_elem.set('CONTENT', cleaned)
                modified += 1

    tree.write(alto_path, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    return modified


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply CATMuS post-processing to ALTO XML files.")
    parser.add_argument("input_dir", help="Directory containing ALTO XML prediction files")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: in-place modification)")
    parser.add_argument("--json-dir", "-j", default=None,
                        help="Directory containing catmus-*.json files (default: same as this script)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output) if args.output else None
    json_dir = Path(args.json_dir) if args.json_dir else Path(__file__).parent

    # Validate
    if not input_dir.is_dir():
        print(f"❌ Input directory not found: {input_dir}")
        return

    json_files = list(json_dir.glob("catmus-*.json"))
    if not json_files:
        print(f"❌ No catmus-*.json files found in: {json_dir}")
        return

    # Build whitelist
    whitelist = _build_catmus_whitelist(json_dir)
    print(f"✓ Whitelist built from {len(json_files)} JSON file(s) — {len(whitelist)} allowed characters")

    # Prepare output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all XML files
    xml_files = sorted(glob.glob(str(input_dir / "**" / "*.xml"), recursive=True))
    if not xml_files:
        print(f"⚠ No XML files found in {input_dir}")
        return

    print(f"Processing {len(xml_files)} file(s)...\n")

    total_modified = 0
    for xml_path in xml_files:
        target = xml_path
        if output_dir:
            rel = Path(xml_path).relative_to(input_dir)
            target = str(output_dir / rel)
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(xml_path, target)

        try:
            n = clean_alto_file(target, whitelist)
            status = f"{n} strings modified" if n else "no changes"
            print(f"  {'→' if output_dir else '✎'} {Path(xml_path).name}  ({status})")
            total_modified += n
        except Exception as e:
            print(f"  ⚠ Error on {Path(xml_path).name}: {e}")

    print(f"\n✅ Done — {total_modified} total string(s) modified across {len(xml_files)} file(s)")


if __name__ == "__main__":
    main()
