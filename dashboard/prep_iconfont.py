"""
Rebuild the self-hosted icon font used by the dashboard.

Downloads Material Symbols Rounded (Apache-2.0, Google), subsets it to just the
~21 glyphs app.py references, compresses to woff2, and prints the base64 blob to
paste into `_ICON_B64` in dashboard/app.py.

Why: Streamlit Cloud / offline demos can't rely on a Google Fonts round-trip for
the icon font — a missed request leaves raw ligature text ("dashboard", "info")
on screen. A 21-glyph subset is ~2.8 KB, so we inline it.

    pip install fonttools brotli
    python -m dashboard.prep_iconfont
"""

from __future__ import annotations

import base64
import io
import urllib.request

# name -> Unicode codepoint  (keep in sync with _ICON_CP in app.py)
CODEPOINTS = {
    "blur_on": 0xE3A5, "dashboard": 0xE871, "satellite_alt": 0xEB3A, "waves": 0xE176,
    "water": 0xF084, "stacked_line_chart": 0xF22B, "scatter_plot": 0xE268,
    "grid_view": 0xE9B0, "compare_arrows": 0xE915, "balance": 0xEAF6,
    "timeline": 0xE922, "science": 0xEA4B, "info": 0xE88E, "sailing": 0xE502,
    "tune": 0xE429, "code": 0xE86F, "monitoring": 0xF190, "description": 0xE873,
    "query_stats": 0xE4FC, "open_in_new": 0xE89E, "north_east": 0xF1E1,
}

# weight-400 static instance of Material Symbols Rounded (from the Google Fonts
# css2 API). Bump the vNNN if Google rotates the URL.
TTF_URL = ("https://fonts.gstatic.com/s/materialsymbolsrounded/v372/"
           "syl0-zNym6YjUruM-QrEh7-nyTnjDwKNJ_190FjpZIvDmUSVOK7BDB_Qb9vUSzq3wzLK"
           "-P0J-V_Zs-QtQth3-jOcbTCVpeRL2w5rwZu2rIelXxI.ttf")


def main():
    req = urllib.request.Request(TTF_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()

    from fontTools import subset
    opts = subset.Options()
    opts.set(layout_features=[], name_IDs=[], notdef_outline=True, glyph_names=False)
    font = subset.load_font(io.BytesIO(raw), opts)
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=list(CODEPOINTS.values()))
    sub.subset(font)
    font.flavor = "woff2"
    buf = io.BytesIO()
    font.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()

    print(f"# {len(buf.getvalue()):,} bytes woff2, {len(CODEPOINTS)} glyphs\n")
    print(f'_ICON_B64 = (\n    "{b64}"\n)')


if __name__ == "__main__":
    main()
