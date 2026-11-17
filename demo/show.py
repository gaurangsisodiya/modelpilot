"""Print part of a Markdown or CSV file readably in a terminal, for the demo (demo/modelpilot.tape).

    python3 demo/show.py results/v6_learned/test_comparison.md    # the whole file
    python3 demo/show.py results/v5_live/live_results.md --columns "Setup,Conc.,Req/s"
    python3 demo/show.py results/baselines.csv                    # CSV: one table, numbers to 3 decimals

Tables are column-aligned. --columns keeps only the named columns, in that order. A table still
wider than the terminal is transposed (one row per column).
Text is wrapped to the terminal width, with ** and ` markup removed. --paged splits the output
into screens at paragraph breaks and waits for Enter between them. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import textwrap
from pathlib import Path

BOLD, DIM, CYAN, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[0m"


def plain(text: str) -> str:
    """Drop the inline Markdown that reads badly in a terminal."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)      # links -> their text
    return text.replace("**", "").replace("`", "").replace("\\*", "*")


def bold_markers(lines: list[str]) -> list[str]:
    """Turn the \\0 markers left in place of ** into bold on/off, carrying bold across wrapped lines."""
    out, bold = [], False
    for line in lines:
        parts = line.split("\0")
        text = (BOLD if bold else "") + parts[0]
        for part in parts[1:]:
            bold = not bold
            text += (BOLD if bold else RESET) + part
        out.append(text + (RESET if bold else ""))
    return out


def cells(row: str) -> list[str]:
    return [plain(c.strip()) for c in row.strip().strip("|").split("|")]


def render_table(rows: list[str], width: int, columns: list[str] | None = None) -> list[str]:
    header, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    if columns:
        missing = [c for c in columns if c not in header]
        if missing:
            sys.exit(f"error: no column {missing[0]!r}; columns are: {', '.join(header)}")
        keep = [header.index(c) for c in columns]
        header, body = [header[i] for i in keep], [[r[i] for i in keep] for r in body]
    numeric = [all(re.fullmatch(r"[-$\d.,/%+ ]+", r[i]) for r in body) for i in range(len(header))]
    if len(" | ".join(header)) > width or any(len(" | ".join(r)) > width for r in body):
        # transpose: first column's values become the new header
        new_header = [header[0]] + [r[0] for r in body]
        new_body = [[header[i]] + [r[i] for r in body] for i in range(1, len(header))]
        header, body = new_header, new_body
        numeric = [False] + [True] * (len(header) - 1)
    widths = [max(len(x) for x in col) for col in zip(header, *body)]

    def line(values: list[str]) -> str:
        return "  ".join(v.rjust(w) if num else v.ljust(w) for v, w, num in zip(values, widths, numeric))

    out = [BOLD + line(header) + RESET, DIM + "  ".join("─" * w for w in widths) + RESET]
    return out + [line(r) for r in body]


def render(lines: list[str], width: int, columns: list[str] | None = None) -> list[str]:
    """Render Markdown lines to terminal lines. Blank lines are kept as paragraph breaks."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        if raw.startswith("|"):
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                table.append(lines[i])
                i += 1
            out += render_table(table, width, columns)
            continue
        i += 1
        if not raw.strip():
            if out and out[-1] != "":
                out.append("")
        elif raw.startswith("#"):
            out.append(BOLD + CYAN + plain(raw.lstrip("#").strip()) + RESET)
        else:
            stripped = raw.lstrip()
            indent = " " * (len(raw) - len(stripped))
            bullet = re.match(r"([-*]|\d+\.) ", stripped)
            first = indent + ("• " if bullet and bullet.group(1) in "-*" else bullet.group(0) if bullet else "")
            body = stripped[bullet.end():] if bullet else stripped
            rest = " " * len(first)
            wrapped = textwrap.wrap(plain(body.replace("**", "\0")), width, initial_indent=first,
                                    subsequent_indent=rest) or [""]
            out += bold_markers(wrapped)
    while out and out[-1] == "":
        out.pop()
    return out


def csv_to_markdown(text: str) -> str:
    """A CSV file as one Markdown table, with floats rounded to 3 decimals."""
    def fmt(value: str) -> str:
        try:
            return value if float(value).is_integer() else f"{float(value):.3f}"
        except ValueError:
            return value
    rows = list(csv.reader(text.splitlines()))
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows[1:]]) + "\n"


def sections(text: str, heading: str) -> list[list[str]]:
    """Each section whose heading starts with `heading`, up to the next heading of the same or a higher level."""
    lines = text.splitlines()
    level = len(heading) - len(heading.lstrip("#"))
    found = []
    for start in (i for i, l in enumerate(lines) if l.startswith(heading)):
        end = next((j for j in range(start + 1, len(lines))
                    if re.match(rf"#{{1,{level}}} ", lines[j])), len(lines))
        found.append(lines[start:end])
    if not found:
        sys.exit(f"error: no heading starting with {heading!r}")
    return found


def pages(lines: list[str], height: int) -> list[list[str]]:
    """Split at blank lines so no page is taller than `height`."""
    blocks, current = [], []
    for line in lines + [""]:
        if line == "":
            if current:
                blocks.append(current)
            current = []
        else:
            current.append(line)
    result, page = [], []
    for block in blocks:
        if page and len(page) + 1 + len(block) > height:
            result.append(page)
            page = []
        page += ([""] if page else []) + block
    return result + ([page] if page else [])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--table", metavar="HEADING", nargs="?", const="",
                       help="print only the first table under this heading (no heading: the first in the file)")
    group.add_argument("--section", metavar="HEADING", help="print this section")
    p.add_argument("--columns", help="comma-separated table columns to keep, in this order")
    p.add_argument("--paged", action="store_true", help="one screen at a time; Enter for the next")
    args = p.parse_args(argv)

    size = shutil.get_terminal_size((100, 30))
    width = size.columns - 1
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    text = args.file.read_text()
    if args.file.suffix == ".csv":
        text = csv_to_markdown(text)
    if args.table is not None:
        # the first matching section that has a table
        candidates = sections(text, args.table) if args.table else [text.splitlines()]
        body = next((b for b in candidates if any(l.startswith("|") for l in b)), None)
        if body is None:
            sys.exit(f"error: no table under {args.table!r}")
        start = next(i for i, l in enumerate(body) if l.startswith("|"))
        end = next((j for j in range(start, len(body)) if not body[j].startswith("|")), len(body))
        out = render_table(body[start:end], width, columns)
    else:
        out = render(sections(text, args.section)[0] if args.section else text.splitlines(), width, columns)

    screens = pages(out, size.lines - 2) if args.paged else [out]
    for n, screen in enumerate(screens):
        if n:
            input(DIM + "(Enter for more)" + RESET)
            print("\033[2J\033[H", end="")
        print("\n".join(screen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
