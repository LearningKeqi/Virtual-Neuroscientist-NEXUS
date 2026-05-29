from pathlib import Path
import shutil
import subprocess
from typing import Optional, Union


def svg_to_png(
    svg: Union[str, Path],
    out_dir: Union[str, Path],
    out_name: Optional[str] = None,
    *,
    dpi: int = 300,
) -> Path:
    """Convert an SVG (file path or SVG string) to PNG and save into out_dir."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = Path(svg) if isinstance(svg, (str, Path)) else None
    svg_is_file = (
        svg_path is not None
        and svg_path.suffix.lower() == ".svg"
        and svg_path.exists()
        and svg_path.is_file()
    )

    if out_name is None:
        out_name = f"{svg_path.stem if svg_is_file else 'image'}.png"
    elif not out_name.lower().endswith(".png"):
        out_name = f"{out_name}.png"

    out_path = out_dir / out_name

    # 1) Preferred: cairosvg (pure python)
    try:
        import cairosvg  # type: ignore

        if svg_is_file:
            cairosvg.svg2png(url=str(svg_path), write_to=str(out_path), dpi=dpi)
        else:
            cairosvg.svg2png(
                bytestring=str(svg).encode("utf-8"), write_to=str(out_path), dpi=dpi
            )
        return out_path
    except ModuleNotFoundError:
        return False




"""
Split a vertically long MRIQC-like mosaic PNG into 3 sub-images without
cutting through slices.

Heuristics used (matches your observations):
- The last row (sagittal) has a different height and is separated from the
  axial mosaic above by a thin, full-width, white seam.
- Above the seam, axial rows have consistent tile height; safe cut lines
  coincide with horizontal grid/separator lines (high vertical-gradient rows).

This script:
1) Detects the sagittal seam (full-white horizontal run not at top/bottom).
2) Detects horizontal separator lines in the axial area via vertical-gradient
   energy and infers the row spacing.
3) Picks 2 cut lines from safe candidates to make 3 parts with similar heights.
4) Writes *_part1.png, *_part2.png, *_part3.png next to the input.

"""


import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def _runs_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start,end) runs where mask is True."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for y, v in enumerate(mask.tolist()):
        if v and start is None:
            start = y
        elif (not v) and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def detect_sagittal_seam_y(gray: np.ndarray) -> int:
    """
    Find the y index of the thin full-width white seam above the bottom row.

    Returns seam y (0..H-1). If not found, falls back to the brightest row
    in the lower half of the image.
    """
    h, _w = gray.shape
    white_frac = (gray >= 245).mean(axis=1)
    full_white = white_frac == 1.0
    runs = _runs_from_mask(full_white)

    # Candidate: very short full-white run, not at extreme top/bottom.
    cands = [
        (s, e)
        for (s, e) in runs
        if (e - s + 1) <= 2 and s > int(h * 0.2) and e < int(h * 0.98)
    ]
    if cands:
        s, e = max(cands, key=lambda t: t[0])  # lowest such run
        return (s + e) // 2

    # Fallback: brightest row in lower half
    lo = int(h * 0.5)
    return int(lo + np.argmax(white_frac[lo:]))


def vertical_gradient_energy(gray: np.ndarray) -> np.ndarray:
    """Mean absolute vertical gradient per boundary row y (length H-1)."""
    g = gray.astype(np.int16)
    return np.abs(g[1:] - g[:-1]).mean(axis=1)


def merge_nearby_positions(ys: Sequence[int], tol: int) -> List[int]:
    """Merge y positions within tol pixels by averaging clusters."""
    if not ys:
        return []
    ys2 = sorted(int(y) for y in ys)
    merged: List[int] = []
    cur: List[int] = [ys2[0]]
    for y in ys2[1:]:
        if y - cur[-1] <= tol:
            cur.append(y)
        else:
            merged.append(int(round(sum(cur) / len(cur))))
            cur = [y]
    merged.append(int(round(sum(cur) / len(cur))))
    return merged


def detect_safe_cut_lines(gray: np.ndarray, seam_y: int) -> List[int]:
    """
    Detect safe horizontal cut lines above (and including) the seam.

    Returns a sorted list of candidate y cut positions in (0, seam_y], where
    each y corresponds to a separator/grid line (safe to cut on).
    """
    h, _w = gray.shape
    seam_y = int(np.clip(seam_y, 1, h - 2))
    G = vertical_gradient_energy(gray)

    # Strong separators in axial area.
    Gax = G[: max(1, seam_y - 1)]
    thr = float(np.quantile(Gax, 0.99))
    peaks = (np.where(Gax >= thr)[0] + 1).astype(int).tolist()
    peaks = merge_nearby_positions(peaks, tol=6)
    peaks = [y for y in peaks if 5 < y < seam_y - 5]
    if len(peaks) < 3:
        # If image differs, broaden the net.
        thr = float(np.quantile(Gax, 0.985))
        peaks = (np.where(Gax >= thr)[0] + 1).astype(int).tolist()
        peaks = merge_nearby_positions(peaks, tol=8)
        peaks = [y for y in peaks if 5 < y < seam_y - 5]

    if len(peaks) < 3:
        # Worst-case fallback: just allow seam only.
        return sorted({seam_y})

    diffs = np.diff(sorted(peaks))
    diffs = diffs[(diffs >= 80) & (diffs <= 250)]  # typical row spacing band
    step = int(round(float(np.median(diffs)))) if len(diffs) else 172

    start0 = min(peaks)
    expected: List[int] = []
    y = start0
    while y < seam_y - 5:
        expected.append(int(y))
        y += step

    # Refine each expected boundary to a local maximum in G.
    refined: List[int] = []
    for y in expected:
        lo = max(1, y - 10)
        hi = min(len(G) - 2, y + 10)
        yy = int(lo + np.argmax(G[lo : hi + 1]))
        if 5 < yy < seam_y - 2:
            refined.append(yy)

    refined.append(seam_y)
    refined = merge_nearby_positions(refined, tol=8)
    refined = [y for y in refined if 5 < y <= seam_y]
    return sorted(set(refined))


def choose_best_2_cuts(
    candidates: Sequence[int], h: int, seam_y: int, min_gap: int = 50
) -> Tuple[int, int]:
    """
    Choose 2 cut lines from candidates to make 3 parts close to equal height.
    Constrain second cut to be <= seam_y (never cut through sagittal row).
    """
    candidates = sorted(set(int(y) for y in candidates))
    candidates = [y for y in candidates if 10 < y < seam_y]
    if len(candidates) < 2:
        # If we only have seam, split at ~1/3 and ~2/3 with seam constraint.
        c1 = int(round(h / 3))
        c2 = min(int(round(2 * h / 3)), seam_y)
        return max(1, c1), max(c1 + 1, c2)

    target = h / 3.0
    best = None  # (obj, c1, c2)
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1 :]:
            if c2 <= c1 + min_gap:
                continue
            if c2 > seam_y:
                continue
            h1, h2, h3 = c1, (c2 - c1), (h - c2)
            obj = max(abs(h1 - target), abs(h2 - target), abs(h3 - target))
            if best is None or obj < best[0]:
                best = (obj, c1, c2)
    if best is None:
        # fallback: pick two farthest-apart candidates
        return candidates[len(candidates) // 3], candidates[(2 * len(candidates)) // 3]
    return best[1], best[2]


def split_image_into_3(
    in_path: Path, out_dir: Optional[Path] = None
) -> Tuple[Path, Path, Path]:
    im = Image.open(in_path)
    gray = np.array(im.convert("L"))
    h, w = gray.shape

    seam_y = detect_sagittal_seam_y(gray)
    candidates = detect_safe_cut_lines(gray, seam_y=seam_y)
    c1, c2 = choose_best_2_cuts(candidates, h=h, seam_y=seam_y)

    out_dir = out_dir or in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out1 = out_dir / f"{in_path.stem}_part1{in_path.suffix}"
    out2 = out_dir / f"{in_path.stem}_part2{in_path.suffix}"
    out3 = out_dir / f"{in_path.stem}_part3{in_path.suffix}"

    im.crop((0, 0, w, c1)).save(out1)
    im.crop((0, c1, w, c2)).save(out2)
    im.crop((0, c2, w, h)).save(out3)
    return out1, out2, out3


def split_mriqc_long_png(input_png: PathLike) -> Tuple[Path, Path, Path]:
    """
    Public API: split one long mosaic PNG into 3 parts.

    - **Input**: path to the original PNG (str or Path)
    - **Output**: writes 3 PNGs into the same folder as the input:
      `*_part1.png`, `*_part2.png`, `*_part3.png`
    - **Return**: (out1, out2, out3) as resolved Paths
    """
    in_path = Path(input_png).expanduser().resolve()
    out1, out2, out3 = split_image_into_3(in_path, out_dir=in_path.parent)
    return [str(out1), str(out2), str(out3)]

