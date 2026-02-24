import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree as ET
from scipy import stats


# ---------------------------------------------------------------------------
# ALTO text extraction
# ---------------------------------------------------------------------------

def extract_text_from_alto(alto_path: str) -> str:
    """Extract transcribed text from an ALTO XML file."""
    try:
        tree = ET.parse(alto_path)
        root = tree.getroot()
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
        strings = root.findall('.//alto:String', ns)
        texts = [s.get('CONTENT', '') for s in strings if s.get('CONTENT')]
        return ' '.join(texts) if texts else ''
    except Exception as e:
        print(f"  ⚠️  Could not read {alto_path}: {e}")
        return ''


# ---------------------------------------------------------------------------
# Character sets loading
# ---------------------------------------------------------------------------

def load_charset(json_path: str) -> set:
    """Load the set of characters from a CATMuS JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    charset = set()
    for entry in data.get('characters', []):
        char = entry.get('character', '')
        if char:
            charset.update(char)  # an entry may be a multi-codepoint cluster
    return charset


def load_charsets_from_folder(folder: str) -> dict:
    """
    Load the three CATMuS charsets from a folder.
    Expected files: catmus-combining.json, catmus-medieval.json, catmus-superscript.json
    Returns a dict with keys: combining, medieval, superscript, merged.
    """
    folder = Path(folder)
    expected = {
        'combining':   'catmus-combining.json',
        'medieval':    'catmus-medieval.json',
        'superscript': 'catmus-superscript.json',
    }
    charsets = {}
    for key, filename in expected.items():
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected charset file not found: {path}")
        charsets[key] = load_charset(str(path))

    charsets['merged'] = charsets['combining'] | charsets['medieval'] | charsets['superscript']
    return charsets


# ---------------------------------------------------------------------------
# Density computation
# ---------------------------------------------------------------------------

def compute_density(text: str, charset: set) -> float:
    """Return the proportion of characters in text that belong to charset."""
    if not text:
        return 0.0
    count = sum(1 for ch in text if ch in charset)
    return count / len(text)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def fmt(value: float) -> str:
    """Format a float rounded to two decimal places."""
    return f"{value:.2f}"


def scatter_with_regression(ax, x: np.ndarray, y: np.ndarray,
                             metric_name: str, charset_name: str, color: str):
    """
    Scatter plot with regression line and annotation showing R², p-value and slope,
    all rounded to two decimal places.
    """
    ax.scatter(x, y, alpha=0.5, s=20, color=color, edgecolors='none')

    if len(x) >= 3 and x.std() > 0:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        r2 = r ** 2
        x_line = np.linspace(x.min(), x.max(), 200)
        label = f"R²={fmt(r2)}  p={fmt(p)}  slope={fmt(slope)}"
        ax.plot(x_line, slope * x_line + intercept, color=color,
                linewidth=1.5, label=label)
        ax.legend(fontsize=8)

    ax.set_xlabel(f'{charset_name} density', fontsize=9)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.set_title(f'{metric_name} vs {charset_name} density', fontsize=10)
    ax.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Correlation between special character density and CER/WER (DocWorkflow)'
    )
    parser.add_argument('--scores',   required=True,
                        help='DocWorkflow score CSV (scores_per_page.csv or scores_all_pages.csv)')
    parser.add_argument('--gt-dir',   required=True,
                        help='Root directory containing ground-truth ALTO XML files')
    parser.add_argument('--charsets', required=True,
                        help='Folder containing the three CATMuS JSON charset files')
    parser.add_argument('--output',   default='.',
                        help='Output directory for figures and enriched CSV (default: .)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load charsets ---
    print("📂 Loading CATMuS charsets…")
    charsets = load_charsets_from_folder(args.charsets)
    print(f"   combining={len(charsets['combining'])} | "
          f"medieval={len(charsets['medieval'])} | "
          f"superscript={len(charsets['superscript'])} | "
          f"merged={len(charsets['merged'])}")

    # --- Load scores ---
    print(f"\n📊 Loading scores: {args.scores}")
    df = pd.read_csv(args.scores)
    print(f"   {len(df)} rows, columns: {list(df.columns)}")

    required = {'page', 'cer', 'wer'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    hierarchical = 'document' in df.columns
    gt_root = Path(args.gt_dir)

    # --- Extract GT text and compute densities ---
    print("\n🔍 Extracting GT texts and computing densities…")
    rows = []
    missing_files = 0

    for _, row in df.iterrows():
        page_stem = row['page']

        if hierarchical:
            xml_path = gt_root / row['document'] / f"{page_stem}.xml"
        else:
            xml_path = gt_root / f"{page_stem}.xml"

        if not xml_path.exists():
            alt = gt_root / f"{page_stem}.xml"
            if alt.exists():
                xml_path = alt
            else:
                missing_files += 1
                continue

        text = extract_text_from_alto(str(xml_path))

        entry = {
            'page':     page_stem,
            'cer':      float(row['cer']),
            'wer':      float(row['wer']),
            'text_len': len(text),
        }
        for cs_name, charset in charsets.items():
            entry[f'density_{cs_name}'] = compute_density(text, charset)

        rows.append(entry)

    if missing_files:
        print(f"  ⚠️  {missing_files} XML files not found (skipped)")

    result_df = pd.DataFrame(rows)
    print(f"  ✓ {len(result_df)} pages processed")

    if result_df.empty:
        print("❌ No data available — check your paths.")
        return

    # Save enriched CSV
    enriched_csv = output_dir / 'scores_with_density.csv'
    result_df.to_csv(enriched_csv, index=False)
    print(f"\n💾 Enriched CSV saved: {enriched_csv}")

    # --- Figures ---
    metrics  = ['cer', 'wer']
    colors   = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    # Figure 1: three separate charsets (3 columns × 2 rows = 6 subplots)
    print("\n🎨 Generating figures…")
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig1.suptitle(
        'CER / WER vs special character density — separate charsets',
        fontsize=13, fontweight='bold'
    )

    for col, cs in enumerate(['combining', 'medieval', 'superscript']):
        for row_idx, metric in enumerate(metrics):
            x = result_df[f'density_{cs}'].values
            y = result_df[metric].values
            scatter_with_regression(axes1[row_idx, col], x, y,
                                    metric.upper(), cs, colors[col])

    fig1_path = output_dir / 'figure1_separate_charsets.png'
    fig1.savefig(fig1_path, dpi=150)
    print(f"  ✓ {fig1_path}")

    # Figure 2: merged charset (2 metrics side by side)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig2.suptitle(
        'CER / WER vs special character density — merged charset',
        fontsize=13, fontweight='bold'
    )

    for idx, metric in enumerate(metrics):
        x = result_df['density_merged'].values
        y = result_df[metric].values
        scatter_with_regression(axes2[idx], x, y, metric.upper(), 'merged', colors[3])

    fig2_path = output_dir / 'figure2_merged_charset.png'
    fig2.savefig(fig2_path, dpi=150)
    print(f"  ✓ {fig2_path}")

    # --- Correlation summary table ---
    print("\n📈 Correlation summary (Pearson):")
    header = (f"  {'Charset':<14} "
              f"{'CER R²':>8} {'CER p':>8} {'CER slope':>10}  "
              f"{'WER R²':>8} {'WER p':>8} {'WER slope':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cs in ['combining', 'medieval', 'superscript', 'merged']:
        col = f'density_{cs}'
        x = result_df[col].values
        if result_df[col].std() == 0:
            print(f"  {cs:<14} {'n/a':>8} {'n/a':>8} {'n/a':>10}  {'n/a':>8} {'n/a':>8} {'n/a':>10}")
            continue
        sl_c, _, r_c, p_c, _ = stats.linregress(x, result_df['cer'].values)
        sl_w, _, r_w, p_w, _ = stats.linregress(x, result_df['wer'].values)
        print(f"  {cs:<14} "
              f"{fmt(r_c**2):>8} {fmt(p_c):>8} {fmt(sl_c):>10}  "
              f"{fmt(r_w**2):>8} {fmt(p_w):>8} {fmt(sl_w):>10}")

    print("\n✅ Done.")


if __name__ == '__main__':
    main()