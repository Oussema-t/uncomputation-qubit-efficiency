#!/usr/bin/env python3
"""Build a single self-contained HTML file from index.html + app.js.

`index.html` loads its logic with `<script src="app.js">`, so the two files must
travel together. Downloading `index.html` on its own yields a page with no
circuits and no numbers, which looks like a broken demo rather than a missing
file. This script inlines the JavaScript so one file works alone.

Run after editing app.js:

    python simulation/build_standalone.py
"""

from __future__ import annotations

import pathlib
import sys

SCRIPT_TAG = '<script src="app.js"></script>'

BANNER = (
    "<!-- Self-contained build: simulation/app.js is inlined below, so this\n"
    "     single file works on its own. Do not edit it directly -- edit app.js\n"
    "     and regenerate:  python simulation/build_standalone.py            -->\n"
)


def build(directory: pathlib.Path) -> pathlib.Path:
    """Inline app.js into index.html and write standalone.html.

    Raises:
        SystemExit: If an input is missing or the inlining would produce broken
            HTML -- a loud failure is better than shipping a silently dead page.
    """
    source = directory / "index.html"
    script = directory / "app.js"
    target = directory / "standalone.html"

    for path in (source, script):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    html = source.read_text(encoding="utf-8")
    js = script.read_text(encoding="utf-8")

    if SCRIPT_TAG not in html:
        raise SystemExit(
            f"could not find {SCRIPT_TAG!r} in {source}; the page layout changed"
        )
    # A literal "</script>" inside the JS would terminate the inlined block
    # early and produce a page that half-works. Refuse rather than emit that.
    if "</script>" in js:
        raise SystemExit(
            f"{script} contains a literal '</script>'; inlining would break the "
            f"page. Split the string (e.g. '<\\/script>') before rebuilding."
        )

    target.write_text(
        html.replace(SCRIPT_TAG, BANNER + "<script>\n" + js + "\n</script>"),
        encoding="utf-8",
    )
    return target


def main() -> int:
    target = build(pathlib.Path(__file__).resolve().parent)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
