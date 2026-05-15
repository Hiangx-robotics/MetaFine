#!/usr/bin/env python3
"""Auto-derive ``assets/<obj>/capabilities.json`` for every asset.

Inference rules (conservative — under-tag rather than over-tag, then refine by
hand):

* Each part listed under ``grasp_parts`` in ``grasp_poses.json`` is
  ``graspable`` — that is the definitional capability that part already has.
* Parts whose linked URDF joint is ``revolute`` or ``continuous`` are tagged
  by semantic name:
    - "lid"/"door" → ``openable``
    - "knob"/"cap"/"handle" → ``rotatable``
    - otherwise → ``rotatable`` (fallback for unlabeled revolute joints)
* Parts whose linked URDF joint is ``prismatic`` → ``slidable``.
* Parts whose semantic name contains "button" or "switch" → ``pressable``.
* Parts whose semantic name contains "body" → ``liftable`` and ``placeable``.

The mapping from a ``grasp_parts`` key (e.g. ``"cap"``) to a URDF link is done
by matching the visual / semantic name. When a match is ambiguous we keep all
candidate affordances. A ``review_needed: true`` flag is emitted to mark each
auto-generated file so a human can refine it.

Run from repo root::

    python utils/derive_capabilities.py --assets-dir assets/
    python utils/derive_capabilities.py --assets-dir assets/ --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

ATOMIC_AFFORDANCES = (
    "graspable",
    "rotatable",
    "slidable",
    "pressable",
    "openable",
    "liftable",
    "insertable",
    "flippable",
    "placeable",
    "stackable",
    "drawable",
)


def parse_semantics(path: Path) -> Dict[str, Tuple[str, str]]:
    """Map link_name → (joint_type_token, semantic_label).

    Example line: ``link_0 hinge rotation_lid`` → ``{"link_0": ("hinge",
    "rotation_lid")}``. Unknown formats are skipped silently.
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) >= 3:
            out[parts[0]] = (parts[1], parts[2])
    return out


def parse_urdf(
    path: Path,
) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, Tuple[str, str]],
    Dict[str, Tuple[Optional[float], Optional[float]]],
    Dict[str, List[Tuple[Optional[str], Tuple[float, float, float]]]],
]:
    """Parse the URDF and return four dicts:

    * ``link_visuals``: link_name → set of <visual name=""> attributes
    * ``link_joint``: child_link_name → (joint_name, joint_type)
    * ``joint_limits``: joint_name → (lower, upper)  (``None`` when absent)
    * ``link_meshes``: link_name → list of (mesh_filename, visual_origin_xyz)
      collected across every ``<visual>`` element on the link. Used by
      :func:`_compute_link_bbox` to derive a coarse axis-aligned bbox.
    """
    link_visuals: Dict[str, Set[str]] = {}
    link_joint: Dict[str, Tuple[str, str]] = {}
    joint_limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    link_meshes: Dict[str, List[Tuple[Optional[str], Tuple[float, float, float]]]] = {}

    tree = ET.parse(path)
    root = tree.getroot()
    for link in root.findall("link"):
        name = link.get("name")
        if name is None:
            continue
        vnames: Set[str] = set()
        meshes: List[Tuple[Optional[str], Tuple[float, float, float]]] = []
        for vis in link.findall("visual"):
            vn = vis.get("name")
            if vn:
                vnames.add(vn)
            origin_xyz = (0.0, 0.0, 0.0)
            origin = vis.find("origin")
            if origin is not None and origin.get("xyz"):
                try:
                    parts = origin.get("xyz").split()
                    if len(parts) == 3:
                        origin_xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
                except Exception:
                    pass
            mesh = vis.find("geometry/mesh") if vis.find("geometry") is not None else None
            mesh_file = mesh.get("filename") if mesh is not None else None
            meshes.append((mesh_file, origin_xyz))
        link_visuals[name] = vnames
        link_meshes[name] = meshes

    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        jname = joint.get("name", "")
        child = joint.find("child")
        if child is None:
            continue
        clink = child.get("link")
        if clink:
            link_joint[clink] = (jname, jtype)
        limit = joint.find("limit")
        lower = upper = None
        if limit is not None:
            try:
                if limit.get("lower") is not None:
                    lower = float(limit.get("lower"))
                if limit.get("upper") is not None:
                    upper = float(limit.get("upper"))
            except Exception:
                pass
        joint_limits[jname] = (lower, upper)

    return link_visuals, link_joint, joint_limits, link_meshes


def _compute_link_bbox(
    meshes: List[Tuple[Optional[str], Tuple[float, float, float]]],
    asset_dir: Path,
) -> Optional[Dict[str, List[float]]]:
    """Coarse axis-aligned bbox of all visuals on a link, in the link frame.

    Loads each mesh via trimesh, translates by its <visual><origin xyz=""/>,
    and aggregates min/max. Returns ``None`` if no usable mesh was found —
    handles per-mesh errors quietly so a single bad file does not abort the
    whole asset.
    """
    try:
        import trimesh  # type: ignore[import-not-found]
    except ImportError:
        return None

    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    found = False
    for mesh_file, origin in meshes:
        if not mesh_file:
            continue
        full = asset_dir / mesh_file
        if not full.exists():
            continue
        try:
            mesh_obj = trimesh.load(str(full), force="mesh", process=False)
            verts = np.asarray(mesh_obj.vertices, dtype=float)
        except Exception:
            continue
        if verts.size == 0:
            continue
        verts = verts + np.asarray(origin, dtype=float)[None, :]
        lo = np.minimum(lo, verts.min(axis=0))
        hi = np.maximum(hi, verts.max(axis=0))
        found = True

    if not found:
        return None
    return {
        "min": [float(v) for v in lo],
        "max": [float(v) for v in hi],
    }


# Map each *canonical* part concept to its surface aliases (case-insensitive
# substring match). Both ``_match_link_for_part`` and
# ``_affordances_from_joint_and_label`` route every free-form label/part name
# through :func:`_canonical_concept` first so synonyms unify naturally —
# e.g. a grasp_parts key ``cap`` and a semantic label ``rotation_lid`` both
# collapse to the canonical ``lid`` and therefore match.
SYNONYMS: Dict[str, List[str]] = {
    "lid":    ["lid", "cap", "cover", "top"],
    "door":   ["door", "gate"],
    "knob":   ["knob", "dial"],
    "drawer": ["drawer", "slider"],
    "handle": ["handle", "grip"],
    "button": ["button", "switch", "key"],
    "body":   ["body", "base", "trunk", "main"],
}


def _canonical_concept(label: str) -> Optional[str]:
    """Map a free-form label to a canonical part concept via :data:`SYNONYMS`.

    Returns ``None`` when no alias matches; callers fall back to the literal
    label in that case.
    """
    lc = label.lower()
    for canon, aliases in SYNONYMS.items():
        for alias in aliases:
            if alias in lc:
                return canon
    return None


def _affordances_from_joint_and_label(joint_type: str, label: str) -> Set[str]:
    """Return joint-driven affordances given URDF joint type and semantic label.

    Uses the canonical concept resolved from the label so synonyms (e.g. a
    bottle "cap" on a "rotation_lid" link) unify naturally.
    """
    canon = _canonical_concept(label) or label.lower()
    out: Set[str] = set()

    if joint_type in ("revolute", "continuous"):
        if canon in ("lid",):
            # A lid that rotates opens (hinge lid → openable). When the
            # joint is "continuous" (no limits, i.e. a screw cap), also
            # mark it rotatable since the cap can spin freely.
            out.add("openable")
            if joint_type == "continuous":
                out.add("rotatable")
        elif canon == "door":
            out.add("openable")
        elif canon == "knob":
            out.add("rotatable")
        elif canon == "handle":
            # Handles on a revolute joint are typically door handles (latch)
            # which both rotate and open.
            out.update(("rotatable", "openable"))
        elif canon == "button":
            # A button on a revolute joint (toggle/press hinge) is press-only;
            # tagging it rotatable would let pure_rotate show up as
            # applicable, which is semantically wrong.
            pass
        else:
            out.add("rotatable")
    elif joint_type == "prismatic":
        out.add("slidable")

    if canon == "button":
        out.add("pressable")
    if canon == "body":
        out.update(("liftable", "placeable"))

    return out


def _match_link_for_part(
    part_name: str,
    semantics: Dict[str, Tuple[str, str]],
    link_visuals: Dict[str, Set[str]],
) -> Optional[str]:
    """Find the URDF link hosting a given grasp_parts key.

    The match cascade goes from strictest to loosest, and falls back to a
    synonym-aware comparison at each level so that, for example, a
    ``grasp_parts`` entry called ``cap`` lines up with a semantic label
    ``rotation_lid`` (both resolve to the canonical concept ``lid``).
    """
    pn = part_name.lower()
    pn_canon = _canonical_concept(pn)

    # 1. exact match on semantic label
    for link, (_jt, label) in semantics.items():
        if label.lower() == pn:
            return link
    # 2. semantic label contains the part name (e.g. "rotation_lid" matches "lid")
    for link, (_jt, label) in semantics.items():
        if pn in label.lower():
            return link
    # 3. canonical-concept match on semantic label (synonyms).
    if pn_canon is not None:
        for link, (_jt, label) in semantics.items():
            if _canonical_concept(label) == pn_canon:
                return link
    # 4. fall back to URDF visual element names (e.g. visual name "lid-8")
    for link, vnames in link_visuals.items():
        for vn in vnames:
            if pn in vn.lower():
                return link
    # 5. canonical-concept match on visual names.
    if pn_canon is not None:
        for link, vnames in link_visuals.items():
            for vn in vnames:
                if _canonical_concept(vn) == pn_canon:
                    return link
    return None


def derive_for_asset(asset_dir: Path) -> Optional[Dict]:
    grasp_poses_path = asset_dir / "grasp_poses.json"
    urdf_path = asset_dir / "mobility.urdf"
    if not grasp_poses_path.exists() or not urdf_path.exists():
        return None

    grasp = json.loads(grasp_poses_path.read_text(encoding="utf-8"))
    grasp_parts = grasp.get("grasp_parts", {})
    if not grasp_parts:
        return None

    semantics = parse_semantics(asset_dir / "semantics.txt")
    link_visuals, link_joint, joint_limits, link_meshes = parse_urdf(urdf_path)
    # Cache per-link bboxes so the same link shared by multiple parts only
    # gets its meshes loaded once.
    bbox_cache: Dict[str, Optional[Dict[str, List[float]]]] = {}

    parts_out: Dict[str, Dict] = {}
    for part_name in grasp_parts.keys():
        affordances: Set[str] = {"graspable"}
        link = _match_link_for_part(part_name, semantics, link_visuals)

        joint_name: Optional[str] = None
        joint_type: Optional[str] = None
        semantic_label: Optional[str] = None

        if link is not None:
            joint_name, joint_type = link_joint.get(link, (None, None))
            semantic_label = semantics.get(link, (None, None))[1]
            if joint_type:
                affordances.update(
                    _affordances_from_joint_and_label(joint_type, semantic_label or part_name)
                )
            else:
                affordances.update(
                    _affordances_from_joint_and_label("", semantic_label or part_name)
                )
                affordances.update(("liftable", "placeable"))
        else:
            affordances.update(
                _affordances_from_joint_and_label("", part_name)
            )

        part_record: Dict = {
            "affordances": sorted(affordances),
        }
        if link:
            part_record["link"] = link
        if joint_name:
            part_record["joint"] = joint_name
        if joint_type:
            part_record["joint_type"] = joint_type
        if semantic_label:
            part_record["semantic_label"] = semantic_label

        # Joint range (only meaningful for revolute/continuous/prismatic).
        if joint_name and joint_name in joint_limits:
            lower, upper = joint_limits[joint_name]
            if lower is not None or upper is not None:
                part_record["joint_range"] = [
                    lower if lower is not None else 0.0,
                    upper if upper is not None else 0.0,
                ]

        # Per-part bbox (computed in the link's local frame).
        if link is not None:
            if link not in bbox_cache:
                bbox_cache[link] = _compute_link_bbox(link_meshes.get(link, []), asset_dir)
            bbox = bbox_cache[link]
            if bbox is not None:
                part_record["bbox_local"] = bbox

        parts_out[part_name] = part_record

    return {
        "object_name": asset_dir.name,
        "parts": parts_out,
        "source": "auto-derived by utils/derive_capabilities.py",
        "review_needed": True,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Derive capabilities.json per asset.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print derived capabilities to stdout; do not write files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing capabilities.json (otherwise skip).")
    args = parser.parse_args(argv)

    if not args.assets_dir.exists():
        print(f"assets dir not found: {args.assets_dir}", file=sys.stderr)
        return 2

    n_done = 0
    n_skipped = 0
    n_existing = 0
    for asset_dir in sorted(p for p in args.assets_dir.iterdir() if p.is_dir()):
        out_path = asset_dir / "capabilities.json"
        if out_path.exists() and not args.overwrite and not args.dry_run:
            n_existing += 1
            continue

        capabilities = derive_for_asset(asset_dir)
        if capabilities is None:
            n_skipped += 1
            print(f"  skip {asset_dir.name}: no grasp_poses.json / mobility.urdf")
            continue

        if args.dry_run:
            print(f"--- {asset_dir.name} ---")
            print(json.dumps(capabilities, indent=2, ensure_ascii=False))
        else:
            out_path.write_text(
                json.dumps(capabilities, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"  wrote {out_path}")
        n_done += 1

    print(
        f"\nSummary: wrote {n_done} files, "
        f"skipped {n_skipped} non-assets, "
        f"preserved {n_existing} existing (use --overwrite to replace)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
