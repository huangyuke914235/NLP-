"""Capture a live Windows ConPTY screen, including its raw VT stream, as a PNG."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
import pyte
from winpty import Backend, PtyProcess

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 85, 27


def run_terminal(argv):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    live = PtyProcess.spawn(
        argv, cwd=str(ROOT), env=env, dimensions=(HEIGHT, WIDTH), backend=Backend.ConPTY
    )
    screen = pyte.Screen(WIDTH, HEIGHT)
    terminal = pyte.Stream(screen)
    raw = []
    while True:
        try:
            chunk = live.read(4096)
        except EOFError:
            break
        raw.append(chunk)
        terminal.feed(chunk)
    if live.wait() != 0:
        raise RuntimeError("Terminal command failed:\n" + "\n".join(screen.display))
    lines = [line.rstrip() for line in screen.display]
    while lines and not lines[-1]:
        lines.pop()
    return lines, "".join(raw)


def save_image(lines, title, destination):
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    font = ImageFont.truetype(str(fonts / "consola.ttf"), 23)
    label_font = ImageFont.truetype(str(fonts / "consolab.ttf"), 21)
    padding, line_height, header = 30, 29, 55
    width = round(font.getlength("M") * WIDTH) + 2 * padding
    image = Image.new("RGB", (width, header + 2 * padding + len(lines) * line_height), "#161D2B")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, header), fill="#263751")
    draw.text((padding, 15), title, fill="#E8F1FE", font=label_font)
    for index, line in enumerate(lines):
        draw.text((padding, header + padding + index * line_height), line, fill="#F1F6FF", font=font)
    image.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("data", "paper"))
    parser.add_argument("--out", type=Path, required=True, help="PNG path, with a companion screen log and VT recording")
    parser.add_argument("--author-repo", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--compact", action="store_true", help="summarize the author scorer's JSON on its live terminal")
    args = parser.parse_args()
    script = "evidence/show_review_terminals.py"
    argv = [sys.executable, "-u", script, args.case]
    if args.case == "paper":
        if args.author_repo is None or args.replay is None:
            parser.error("paper requires --author-repo and --replay")
        argv.extend(["--author-repo", str(args.author_repo.resolve()), "--replay", str(args.replay.resolve())])
        if args.compact:
            argv.append("--compact")
    snapshot, stream = run_terminal(argv)
    if not snapshot:
        raise ValueError("The live terminal produced no visible output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".log").write_text("\n".join(snapshot) + "\n", encoding="utf-8")
    args.out.with_suffix(".vt").write_bytes(stream.encode("utf-8"))
    suffix = " --compact" if args.compact else ""
    save_image(snapshot, f"ConPTY > python {script} {args.case}{suffix}", args.out)
    print(f"Captured {len(snapshot)} screen lines to {args.out}")


if __name__ == "__main__":
    main()
