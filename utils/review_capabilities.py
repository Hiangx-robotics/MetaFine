#!/usr/bin/env python3
"""Interactive review tool for auto-generated ``capabilities.json`` files.

The auto-derive heuristic in :mod:`utils.derive_capabilities` is deliberately
conservative (better to under-tag than to over-tag a part with a capability it
cannot actually fulfil), so every emitted file carries ``review_needed: true``.
This script gives you a quick way to walk those files, inspect the inferred
affordances + joint range + bbox, flip any tag the heuristic got wrong, and
mark the file as reviewed.

Three usage modes:

1. **List**::

       python utils/review_capabilities.py list                    # all assets
       python utils/review_capabilities.py list --review-only      # only review_needed=true
       python utils/review_capabilities.py list bottle             # one asset

2. **Toggle** (non-interactive, scriptable)::

       python utils/review_capabilities.py toggle bottle cap +rotatable
       python utils/review_capabilities.py toggle bottle cap -- -openable
       python utils/review_capabilities.py toggle bottle cap -- +flippable -openable

   Prefix an affordance with ``+`` to add, ``-`` to remove. When any token
   starts with ``-`` (a remove), put ``--`` before the toggles so argparse
   doesn't treat them as flags.

3. **Confirm** — mark ``review_needed: false`` once you're happy::

       python utils/review_capabilities.py confirm bottle
       python utils/review_capabilities.py confirm --all  # batch mark all assets

A small ``walk`` mode (default when called with no arguments) walks every
``review_needed: true`` asset and drops you into a tiny REPL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Mirrors core.skill_registry.AFFORDANCES (kept inline so the tool runs
# without needing the FGManip package on sys.path).
AFFORDANCES = (
    "graspable", "rotatable", "slidable", "pressable",
    "openable", "liftable", "insertable", "flippable",
    "placeable", "stackable", "drawable",
)


# --------------------------------------------------------------------------- #
# IO                                                                          #
# --------------------------------------------------------------------------- #

def _capabilities_path(assets_dir: Path, name: str) -> Path:
    return assets_dir / name / "capabilities.json"


def load_capabilities(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_capabilities(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_assets(assets_dir: Path, review_only: bool = False) -> Iterable[Tuple[str, Path]]:
    for sub in sorted(p for p in assets_dir.iterdir() if p.is_dir()):
        cap = sub / "capabilities.json"
        if not cap.exists():
            continue
        if review_only:
            try:
                data = load_capabilities(cap)
            except Exception:
                continue
            if not data.get("review_needed"):
                continue
        yield sub.name, cap


# --------------------------------------------------------------------------- #
# Display                                                                     #
# --------------------------------------------------------------------------- #

def _format_bbox(bbox: Optional[Dict]) -> str:
    if bbox is None:
        return "—"
    lo, hi = bbox["min"], bbox["max"]
    return f"[{hi[0]-lo[0]:.3f} × {hi[1]-lo[1]:.3f} × {hi[2]-lo[2]:.3f}] m"


def render_capabilities(data: Dict, name: str) -> str:
    lines = [f"\n{name}   (review_needed={data.get('review_needed', False)})"]
    parts = data.get("parts", {})
    if not parts:
        lines.append("  (no parts)")
        return "\n".join(lines)
    for pname, meta in parts.items():
        joint = meta.get("joint")
        jtype = meta.get("joint_type")
        jrange = meta.get("joint_range")
        sem = meta.get("semantic_label", "—")
        bbox = _format_bbox(meta.get("bbox_local"))
        aff = ", ".join(meta.get("affordances", [])) or "(none)"
        joint_str = f"{joint} ({jtype})" if joint else "—"
        if jrange:
            joint_str += f" range [{jrange[0]:.3f}, {jrange[1]:.3f}]"
        lines.append(f"  {pname:24s}  semantic={sem:18s}  joint={joint_str}")
        lines.append(f"  {'':24s}  bbox={bbox}")
        lines.append(f"  {'':24s}  affordances=[{aff}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Mutations                                                                   #
# --------------------------------------------------------------------------- #

def apply_toggles(data: Dict, part_name: str, toggles: List[str]) -> List[str]:
    """Apply +affordance / -affordance toggles to a part. Returns log lines."""
    parts = data.get("parts", {})
    if part_name not in parts:
        raise KeyError(f"part {part_name!r} not in this asset; have {list(parts)}")
    record = parts[part_name]
    current = set(record.get("affordances", []))
    log: List[str] = []
    for tok in toggles:
        if not tok or tok[0] not in "+-":
            raise ValueError(f"toggle {tok!r} must start with + or -")
        op, aff = tok[0], tok[1:]
        if aff not in AFFORDANCES:
            raise ValueError(f"unknown affordance {aff!r}; valid: {list(AFFORDANCES)}")
        if op == "+":
            if aff in current:
                log.append(f"  already had {aff}")
            else:
                current.add(aff)
                log.append(f"  +{aff}")
        else:
            if aff not in current:
                log.append(f"  did not have {aff}")
            else:
                current.discard(aff)
                log.append(f"  -{aff}")
    record["affordances"] = sorted(current)
    return log


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #

def cmd_list(args: argparse.Namespace) -> int:
    assets_dir = args.assets_dir
    if args.names:
        for name in args.names:
            path = _capabilities_path(assets_dir, name)
            if not path.exists():
                print(f"[skip] {name}: no capabilities.json", file=sys.stderr)
                continue
            print(render_capabilities(load_capabilities(path), name))
    else:
        for name, path in iter_assets(assets_dir, review_only=args.review_only):
            print(render_capabilities(load_capabilities(path), name))
    return 0


def cmd_toggle(args: argparse.Namespace) -> int:
    path = _capabilities_path(args.assets_dir, args.object)
    if not path.exists():
        print(f"no capabilities for {args.object}", file=sys.stderr)
        return 2
    data = load_capabilities(path)
    log = apply_toggles(data, args.part, args.toggles)
    save_capabilities(path, data)
    print(f"{args.object}.{args.part}:")
    for line in log:
        print(line)
    print(render_capabilities(data, args.object))
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    if args.all:
        targets = [name for name, _ in iter_assets(args.assets_dir, review_only=True)]
    else:
        targets = args.objects
    if not targets:
        print("nothing to confirm")
        return 0
    for name in targets:
        path = _capabilities_path(args.assets_dir, name)
        if not path.exists():
            print(f"[skip] {name}: no capabilities.json")
            continue
        data = load_capabilities(path)
        if not data.get("review_needed"):
            print(f"[skip] {name}: already confirmed")
            continue
        data["review_needed"] = False
        save_capabilities(path, data)
        print(f"confirmed: {name}")
    return 0


def cmd_walk(args: argparse.Namespace) -> int:
    """Tiny REPL: show one asset, accept `+aff <part>`, `-aff <part>`, n/skip/quit."""
    for name, path in iter_assets(args.assets_dir, review_only=True):
        data = load_capabilities(path)
        print(render_capabilities(data, name))
        while True:
            cmd = input(
                f"\n[{name}] (n)ext (q)uit (c)onfirm  or  '<part> +aff -aff ...': "
            ).strip()
            if cmd in ("", "n", "next"):
                break
            if cmd in ("q", "quit"):
                return 0
            if cmd in ("c", "confirm"):
                data["review_needed"] = False
                save_capabilities(path, data)
                print("  confirmed")
                break
            tokens = cmd.split()
            if len(tokens) < 2:
                print("  format: <part> +aff [-aff] [+aff] ...")
                continue
            try:
                log = apply_toggles(data, tokens[0], tokens[1:])
                save_capabilities(path, data)
                for line in log:
                    print(line)
                print(render_capabilities(data, name))
            except (KeyError, ValueError) as exc:
                print(f"  error: {exc}")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive review of auto-generated capabilities.json files.",
    )
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="Print capabilities for some/all assets.")
    p_list.add_argument("names", nargs="*", help="Asset names; default = all.")
    p_list.add_argument("--review-only", action="store_true",
                        help="Only print assets whose review_needed flag is true.")
    p_list.set_defaults(func=cmd_list)

    p_tog = sub.add_parser("toggle", help="Add/remove affordances on a part (scriptable).")
    p_tog.add_argument("object")
    p_tog.add_argument("part")
    p_tog.add_argument("toggles", nargs="+",
                       help="Each token is +<affordance> or -<affordance>.")
    p_tog.set_defaults(func=cmd_toggle)

    p_ok = sub.add_parser("confirm", help="Set review_needed=false on listed objects.")
    p_ok.add_argument("objects", nargs="*")
    p_ok.add_argument("--all", action="store_true",
                      help="Mark every review_needed=true asset as confirmed.")
    p_ok.set_defaults(func=cmd_confirm)

    p_walk = sub.add_parser("walk", help="Interactive REPL across review_needed assets.")
    p_walk.set_defaults(func=cmd_walk)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # Default: walk every review_needed asset.
        args.command = "walk"
        args.func = cmd_walk
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
