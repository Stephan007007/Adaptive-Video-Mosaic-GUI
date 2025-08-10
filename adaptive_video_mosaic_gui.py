"""
Adaptive Video/GIF Mosaic GUI — with output resolution control

Changes made:
- Added UI controls to set output scaling (%) manually.
- Added an "Auto-adjust resolution" button which analyzes a reference frame and suggests a scale so that tiles are more visually clear.
- The generator now rescales target frames before computing tiles / rendering so the output GIF/MP4 is produced at the requested resolution.
- Caps and safety checks to avoid extreme upscales.

Usage: run the script and use the new "Output scale (%)" control (default 100).
Click "Auto-adjust resolution" to compute a recommended scale based on the reference frame and current tile parameters.

Note: Rescaling increases memory usage and processing time. The automatic suggestion aims to make the typical tile ~48 px high/wide for better visual readability.

Save as: adaptive_video_mosaic_gui_with_resolution_control.py
Run: python adaptive_video_mosaic_gui_with_resolution_control.py
"""
import os
import threading
import time
import math
import random
from functools import lru_cache
from tqdm import tqdm

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw
import numpy as np

# moviepy for writing videos
from moviepy.editor import VideoFileClip, ImageSequenceClip

# -------- file type sets ----------
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTS = {'.gif', '.mp4', '.mov', '.avi', '.webm', '.mkv'}


def is_image_file(path):
    return os.path.splitext(path.lower())[1] in IMAGE_EXTS


def is_video_file(path):
    return os.path.splitext(path.lower())[1] in VIDEO_EXTS


def is_gif_file(path):
    return os.path.splitext(path.lower())[1] == '.gif'


# -------- small helpers ----------
def pil_from_frame(frame_np):
    """Convert an RGB numpy frame (H,W,3) to PIL Image."""
    if frame_np.dtype != np.uint8:
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_np)


def compute_average_color_pil(img, small=32):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > small:
        if w >= h:
            img = img.resize((small, int(round(h * small / w))), Image.Resampling.LANCZOS)
        else:
            img = img.resize((int(round(w * small / h)), small), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return arr.mean(axis=(0, 1))


def average_hash(img, hash_size=8):
    if img.mode != 'L':
        im = img.convert('L')
    else:
        im = img
    im = im.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.uint8)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming_distance(a, b):
    return (a ^ b).bit_count()


# -------- quadtree tiling ----------
def quadtree_tiles(np_img, min_tile=24, max_tile=256, var_threshold=500.0):
    h, w = np_img.shape[:2]
    tiles = []

    def _region_var(x, y, ww, hh):
        reg = np_img[y:y+hh, x:x+ww]
        return float(reg.var(axis=(0, 1)).mean())

    def split(x, y, ww, hh):
        v = _region_var(x, y, ww, hh)
        if ww <= min_tile or hh <= min_tile:
            tiles.append((x, y, ww, hh))
            return
        if (ww > max_tile and hh > max_tile) or (v > var_threshold and ww > min_tile and hh > min_tile):
            w2 = ww // 2
            h2 = hh // 2
            if w2 < 1 or h2 < 1:
                tiles.append((x, y, ww, hh))
                return
            split(x, y, w2, h2)
            split(x + w2, y, ww - w2, h2)
            split(x, y + h2, w2, hh - h2)
            split(x + w2, y + h2, ww - w2, hh - h2)
        else:
            tiles.append((x, y, ww, hh))

    split(0, 0, w, h)
    return tiles


# -------- build source index (images + short videos/gifs) ----------
def build_source_index(folder, thumb_for_avg=32, hash_size=8, max_images=None, max_frames_per_source=6, progress_callback=None):
    """
    Scan source folder. For images: store single frame. For video/gif: sample up to max_frames_per_source frames evenly.
    Returns list of dicts:
      {
        'path': path,
        'is_anim': bool,
        'frames': [PIL Image thumbnails...],
        'frame_avgs': [np.array],
        'avg': np.array(mean across frames),
        'ahash': int (from first frame)
      }
    """
    items = []
    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                all_files.append(os.path.join(root, f))
    if max_images:
        all_files = all_files[:max_images]
    n = len(all_files)
    for i, p in enumerate(all_files):
        try:
            ext = os.path.splitext(p)[1].lower()
            if ext in IMAGE_EXTS:
                with Image.open(p) as img:
                    img = img.convert('RGB')
                    thumb = img.copy()
                    # modest thumbnail to save memory/time
                    thumb.thumbnail((160, 160), Image.Resampling.LANCZOS)
                    fa = compute_average_color_pil(thumb, small=thumb_for_avg)
                    ah = average_hash(thumb.convert('L'), hash_size=hash_size)
                    items.append({'path': p, 'is_anim': False, 'frames': [thumb], 'frame_avgs': [fa], 'avg': fa, 'ahash': ah})
            else:
                # video/gif: sample frames using moviepy (robust for many formats)
                clip = VideoFileClip(p)
                dur = max(clip.duration, 0.001)
                n_samples = min(max(1, int(round(max_frames_per_source))), max_frames_per_source)
                times = [ (dur * (t + 0.5) / n_samples) for t in range(n_samples) ]
                frames = []
                frame_avgs = []
                for tsec in times:
                    try:
                        frame_np = clip.get_frame(min(tsec, dur - 1e-3))
                        pil = pil_from_frame(frame_np)
                        pil.thumbnail((160, 160), Image.Resampling.LANCZOS)
                        fa = compute_average_color_pil(pil, small=thumb_for_avg)
                        frames.append(pil)
                        frame_avgs.append(fa)
                    except Exception:
                        continue
                clip.close()
                if len(frames) == 0:
                    continue
                avg_all = np.mean(frame_avgs, axis=0)
                ah = average_hash(frames[0].convert('L'), hash_size=hash_size)
                items.append({'path': p, 'is_anim': True, 'frames': frames, 'frame_avgs': frame_avgs, 'avg': avg_all, 'ahash': ah})
        except Exception as e:
            print('Skipping', p, 'error:', e)
        if progress_callback and (i % 5 == 0 or i == n-1):
            progress_callback(i + 1, n)
    if progress_callback:
        progress_callback(n, n)
    return items


# -------- rendering helpers ----------
def _render_source_frame_to_region(pil_frame, out_w, out_h, tile_avg, stretch_mode='keep', max_stretch_factor=3.0, force_full_stretch=False):
    s = pil_frame.convert('RGB')
    sw, sh = s.size
    target_aspect = out_w / out_h if out_h != 0 else 1.0
    src_aspect = sw / sh if sh != 0 else 1.0

    if stretch_mode == 'keep':
        if src_aspect > target_aspect:
            new_w = int(round(sh * target_aspect))
            left = max(0, (sw - new_w) // 2)
            s = s.crop((left, 0, left + new_w, sh))
        else:
            new_h = int(round(sw / target_aspect)) if target_aspect != 0 else sh
            top = max(0, (sh - new_h) // 2)
            s = s.crop((0, top, sw, top + new_h))
        s = s.resize((out_w, out_h), Image.Resampling.LANCZOS)
        return s
    elif stretch_mode == 'fit':
        if src_aspect > target_aspect:
            new_w = out_w
            new_h = max(1, int(round(new_w / src_aspect)))
        else:
            new_h = out_h
            new_w = max(1, int(round(new_h * src_aspect)))
        s = s.resize((new_w, new_h), Image.Resampling.LANCZOS)
        bg = Image.new('RGB', (out_w, out_h), tuple(int(max(0, min(255, c))) for c in tile_avg))
        offx = (out_w - new_w) // 2
        offy = (out_h - new_h) // 2
        bg.paste(s, (offx, offy))
        return bg
    else:  # 'stretch' per-axis
        sx = out_w / sw if sw != 0 else 1.0
        sy = out_h / sh if sh != 0 else 1.0
        if sx <= max_stretch_factor and sy <= max_stretch_factor:
            return s.resize((out_w, out_h), Image.Resampling.LANCZOS)
        else:
            if force_full_stretch:
                return s.resize((out_w, out_h), Image.Resampling.LANCZOS)
            sx_l = min(sx, max_stretch_factor)
            sy_l = min(sy, max_stretch_factor)
            new_w = max(1, int(round(sw * sx_l)))
            new_h = max(1, int(round(sh * sy_l)))
            s = s.resize((new_w, new_h), Image.Resampling.LANCZOS)
            bg = Image.new('RGB', (out_w, out_h), tuple(int(max(0, min(255, c))) for c in tile_avg))
            offx = (out_w - new_w) // 2
            offy = (out_h - new_h) // 2
            bg.paste(s, (offx, offy))
            return bg


# -------- group merging and rendering ----------
def compute_groups_for_tiles(tiles, match_idx, src_hashes, src_colors, hash_threshold=6, allow_diagonal=True, gap_tolerance=2, min_tiles_for_merge=3):
    n_src = len(src_hashes)
    cluster_id = list(range(n_src))

    def find(a):
        while cluster_id[a] != a:
            cluster_id[a] = cluster_id[cluster_id[a]]
            a = cluster_id[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            cluster_id[rb] = ra

    if n_src > 1 and hash_threshold >= 0:
        for i in range(n_src):
            for j in range(i + 1, n_src):
                if np.linalg.norm(src_colors[i] - src_colors[j]) > 60:
                    continue
                if hamming_distance(src_hashes[i], src_hashes[j]) <= hash_threshold:
                    union(i, j)
    for i in range(n_src):
        cluster_id[i] = find(i)
    cluster_map = {}
    next_cluster = 0
    for i in range(n_src):
        r = cluster_id[i]
        if r not in cluster_map:
            cluster_map[r] = next_cluster
            next_cluster += 1
    src_cluster = [cluster_map[cluster_id[i]] if n_src > 0 else -1 for i in range(n_src)]

    def rects_connect(a, b, gap, allow_diag):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh
        if (ax2 + gap < bx) or (bx2 + gap < ax) or (ay2 + gap < by) or (by2 + gap < ay):
            return False
        if not allow_diag:
            horiz = (ax2 + gap >= bx and bx2 + gap >= ax) and (max(ay, by) < min(ay2, by2))
            vert = (ay2 + gap >= by and by2 + gap >= ay) and (max(ax, bx) < min(ax2, bx2))
            return horiz or vert
        else:
            return True

    n_tiles = len(tiles)
    processed = [False] * n_tiles
    groups = []
    for i in range(n_tiles):
        if processed[i]:
            continue
        processed[i] = True
        comp = [i]
        queue = [i]
        while queue:
            u = queue.pop()
            for v in range(n_tiles):
                if processed[v]:
                    continue
                if match_idx[u] == -1 or match_idx[v] == -1:
                    continue
                if src_cluster[match_idx[u]] != src_cluster[match_idx[v]]:
                    continue
                if rects_connect(tiles[u], tiles[v], gap_tolerance, allow_diagonal):
                    processed[v] = True
                    queue.append(v)
                    comp.append(v)
        groups.append(comp)
    return groups, src_cluster


# -------- main rendering loop for video frames ----------
def render_frames_for_video(
        frames_np,
        tiles=None,
        src_index=None,
        groups=None,
        src_cluster=None,
        stretch_mode='keep', max_stretch_factor=3.0, force_full_stretch=False,
        merge_area_threshold=0.85, min_group_tiles=3, min_merge_bbox_area=5000,
        animate_source_mode='cycle',
        opacity=0.0,
        progress_callback=None,
        stop_flag=None,
        # added parameters for per-frame tiling
        compute_tiles_per_frame=False,
        min_tile=24, max_tile=256, var_threshold=500.0,
        # clustering params
        hash_threshold=6, allow_diagonal=True, gap_tolerance=2
    ):
    """Render each target frame into a mosaic frame (PIL Images).
    frames_np: list of numpy arrays (RGB)
    If compute_tiles_per_frame True, tiles and groups will be computed per frame (slower but adaptive).
    Returns list of PIL frames.
    """
    if src_index is None:
        src_index = []
    n_src = len(src_index)
    src_colors = np.array([src['avg'] for src in src_index], dtype=np.float32) if n_src > 0 else np.zeros((0,3), dtype=np.float32)
    src_hashes = [src['ahash'] for src in src_index] if n_src > 0 else []

    # If static tiles/groups provided (consistent mode), use as-is. Else compute per-frame inside the loop.
    # A small cache for rendered tile images
    render_cache = {}

    def render_tile_for_source_frame(src_idx, frame_variant_idx, out_w, out_h, tile_avg):
        key = (src_idx, frame_variant_idx, out_w, out_h, stretch_mode, max_stretch_factor, force_full_stretch)
        if key in render_cache:
            return render_cache[key]
        src = src_index[src_idx]
        if src['is_anim']:
            frames = src['frames']
            j = frame_variant_idx % len(frames)
            pil_frame = frames[j]
        else:
            pil_frame = src['frames'][0]
        tile_img = _render_source_frame_to_region(pil_frame, out_w, out_h, tile_avg, stretch_mode, max_stretch_factor, force_full_stretch)
        render_cache[key] = tile_img
        return tile_img

    out_frames = []
    total = len(frames_np)
    for fi, f_np in enumerate(frames_np):
        if stop_flag and stop_flag():
            raise RuntimeError('Stopped by user')
        h, w = f_np.shape[:2]

        # decide tiles/groups for this frame
        if compute_tiles_per_frame:
            tiles_frame = quadtree_tiles(f_np.astype(np.float32), min_tile=min_tile, max_tile=max_tile, var_threshold=var_threshold)
            # compute tile avgs
            tile_avgs = []
            for (x,y,tw,th) in tiles_frame:
                reg = f_np[y:y+th, x:x+tw]
                tile_avgs.append(reg.mean(axis=(0,1)) if reg.size else np.array([0.0,0.0,0.0]))
            # compute match idx by comparing to src_colors
            if src_colors.shape[0] == 0:
                match_idx = [-1]*len(tiles_frame)
            else:
                match_idx = []
                for ta in tile_avgs:
                    diffs = src_colors - ta
                    dists = np.einsum('ij,ij->i', diffs, diffs)
                    match_idx.append(int(np.argmin(dists)))
            # compute groups for this frame
            groups_frame, src_cluster_frame = compute_groups_for_tiles(tiles_frame, match_idx, src_hashes, src_colors, hash_threshold=hash_threshold, allow_diagonal=allow_diagonal, gap_tolerance=gap_tolerance, min_tiles_for_merge=min_group_tiles)
        else:
            tiles_frame = tiles
            match_idx = None  # computed earlier by caller when groups provided
            groups_frame = groups
            src_cluster_frame = src_cluster
            # If groups/frame-mapping wasn't provided, create a simple match list from first-frame mapping
            if groups_frame is None:
                # fallback: compute mapping from this frame to sources
                tile_avgs = []
                if tiles_frame is None:
                    tiles_frame = quadtree_tiles(f_np.astype(np.float32), min_tile=min_tile, max_tile=max_tile, var_threshold=var_threshold)
                for (x,y,tw,th) in tiles_frame:
                    reg = f_np[y:y+th, x:x+tw]
                    tile_avgs.append(reg.mean(axis=(0,1)) if reg.size else np.array([0.0,0.0,0.0]))
                if src_colors.shape[0] == 0:
                    match_idx = [-1]*len(tiles_frame)
                else:
                    match_idx = []
                    for ta in tile_avgs:
                        diffs = src_colors - ta
                        dists = np.einsum('ij,ij->i', diffs, diffs)
                        match_idx.append(int(np.argmin(dists)))
                groups_frame, src_cluster_frame = compute_groups_for_tiles(tiles_frame, match_idx, src_hashes, src_colors, hash_threshold=hash_threshold, allow_diagonal=allow_diagonal, gap_tolerance=gap_tolerance, min_tiles_for_merge=min_group_tiles)
            else:
                # precomputed match_idx can be constructed from groups (one representative per tile)
                # We don't have direct match_idx_ref here (unless provided), so let's compute a match index mapping via avg
                tile_avgs = []
                for (x,y,tw,th) in tiles_frame:
                    reg = f_np[y:y+th, x:x+tw]
                    tile_avgs.append(reg.mean(axis=(0,1)) if reg.size else np.array([0.0,0.0,0.0]))
                if src_colors.shape[0] == 0:
                    match_idx = [-1]*len(tiles_frame)
                else:
                    match_idx = []
                    for ta in tile_avgs:
                        diffs = src_colors - ta
                        dists = np.einsum('ij,ij->i', diffs, diffs)
                        match_idx.append(int(np.argmin(dists)))

        # Prepare variant mapping for animated sources
        src_variant_for_frame = [0]*n_src
        if animate_source_mode == 'cycle':
            for s_i in range(n_src):
                nvar = len(src_index[s_i]['frames'])
                src_variant_for_frame[s_i] = fi % max(1, nvar)

        mosaic = Image.new('RGB', (w, h))

        # iterate groups (or tiles individually if no merging)
        for gi, group in enumerate(groups_frame):
            xs = [tiles_frame[i][0] for i in group]
            ys = [tiles_frame[i][1] for i in group]
            rights = [tiles_frame[i][0] + tiles_frame[i][2] for i in group]
            bottoms = [tiles_frame[i][1] + tiles_frame[i][3] for i in group]
            min_x = min(xs); min_y = min(ys)
            bbox_w = max(rights) - min_x; bbox_h = max(bottoms) - min_y
            area_sum = sum(tiles_frame[i][2]*tiles_frame[i][3] for i in group)
            bbox_area = max(1, bbox_w*bbox_h)
            coverage = area_sum / bbox_area

            # representative tile's matched source (from match_idx)
            rep_tile = group[0]
            rep_match = match_idx[rep_tile] if match_idx is not None else -1
            can_merge = (rep_match != -1 and coverage >= merge_area_threshold and len(group) >= min_group_tiles and bbox_w*bbox_h >= min_merge_bbox_area)

            if can_merge:
                src_idx = rep_match
                if animate_source_mode == 'cycle':
                    variant = src_variant_for_frame[src_idx]
                elif animate_source_mode == 'best':
                    if src_index[src_idx]['is_anim']:
                        region = f_np[min_y:min_y+bbox_h, min_x:min_x+bbox_w]
                        targ_avg = region.mean(axis=(0,1)) if region.size else np.array([0.0,0.0,0.0])
                        frame_avgs = src_index[src_idx]['frame_avgs']
                        dists = [np.linalg.norm(fa - targ_avg) for fa in frame_avgs]
                        variant = int(np.argmin(dists))
                    else:
                        variant = 0
                else:
                    variant = 0

                region = f_np[min_y:min_y+bbox_h, min_x:min_x+bbox_w]
                tile_avg_here = region.mean(axis=(0,1)) if region.size else np.array([0.0,0.0,0.0])
                tile_img = render_tile_for_source_frame(src_idx, variant, bbox_w, bbox_h, tile_avg_here)
                mosaic.paste(tile_img, (min_x, min_y))
            else:
                for ti in group:
                    x,y,tw,th = tiles_frame[ti]
                    region = f_np[y:y+th, x:x+tw]
                    tile_avg_here = region.mean(axis=(0,1)) if region.size else np.array([0.0,0.0,0.0])
                    match = match_idx[ti]
                    if match == -1:
                        tile_img = Image.new('RGB', (tw, th), tuple(int(max(0, min(255, c))) for c in tile_avg_here))
                    else:
                        if animate_source_mode == 'cycle':
                            variant = src_variant_for_frame[match]
                        elif animate_source_mode == 'best' and src_index[match]['is_anim']:
                            frame_avgs = src_index[match]['frame_avgs']
                            dists = [np.linalg.norm(fa - tile_avg_here) for fa in frame_avgs]
                            variant = int(np.argmin(dists))
                        else:
                            variant = 0
                        tile_img = render_tile_for_source_frame(match, variant, tw, th, tile_avg_here)
                    mosaic.paste(tile_img, (x,y))

        orig_pil = Image.fromarray(f_np.astype(np.uint8))
        if opacity > 0:
            blended = Image.blend(mosaic, orig_pil, opacity)
            out_frames.append(blended)
        else:
            out_frames.append(mosaic)

        if progress_callback and (fi % 3 == 0 or fi == total - 1):
            progress_callback(fi + 1, total)

    return out_frames


# ---------- GUI app ----------
class VideoMosaicApp:
    def __init__(self, root):
        self.root = root
        root.title('Adaptive Video & GIF Mosaic (with resolution control)')
        self.src_index = []
        self.stop = False

        frm = ttk.Frame(root, padding=8)
        frm.grid(row=0, column=0, sticky='nsew')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        r = 0
        ttk.Label(frm, text='Target video/GIF:').grid(row=r, column=0, sticky='w')
        self.target_entry = ttk.Entry(frm, width=50)
        self.target_entry.grid(row=r, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_target).grid(row=r, column=2)
        r += 1

        ttk.Label(frm, text='Source folder (images/video/gif):').grid(row=r, column=0, sticky='w')
        self.src_entry = ttk.Entry(frm, width=50)
        self.src_entry.grid(row=r, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_src_and_scan).grid(row=r, column=2)
        r += 1

        ttk.Label(frm, text='Output folder:').grid(row=r, column=0, sticky='w')
        self.out_entry = ttk.Entry(frm, width=50)
        self.out_entry.grid(row=r, column=1, sticky='ew')
        ttk.Button(frm, text='Browse', command=self.browse_out).grid(row=r, column=2)
        r += 1

        # processing options
        self.consistent_tiles_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Compute consistent tiles from reference frame (recommended)', variable=self.consistent_tiles_var).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1

        ttk.Label(frm, text='Reference frame selector:').grid(row=r, column=0, sticky='w')
        self.ref_frame_combo = tk.StringVar(value='median')
        self.ref_frame_box = ttk.Combobox(frm, textvariable=self.ref_frame_combo, state='readonly', values=('first','median','frame N'))
        self.ref_frame_box.grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Max frames to process (0=all):').grid(row=r, column=0, sticky='w')
        self.max_frames = tk.IntVar(value=0)
        ttk.Spinbox(frm, from_=0, to=2000, textvariable=self.max_frames).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Frame step (process every Nth frame):').grid(row=r, column=0, sticky='w')
        self.frame_step = tk.IntVar(value=1)
        ttk.Spinbox(frm, from_=1, to=60, textvariable=self.frame_step).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Output fps (for mp4):').grid(row=r, column=0, sticky='w')
        self.out_fps = tk.IntVar(value=12)
        ttk.Spinbox(frm, from_=1, to=60, textvariable=self.out_fps).grid(row=r, column=1, sticky='w')
        r += 1

        # NEW: output resolution / scale controls
        ttk.Label(frm, text='Output scale (%) — 100 = native size:').grid(row=r, column=0, sticky='w')
        self.out_scale = tk.IntVar(value=100)
        self.out_scale_spin = ttk.Spinbox(frm, from_=10, to=2000, textvariable=self.out_scale)
        self.out_scale_spin.grid(row=r, column=1, sticky='w')
        ttk.Button(frm, text='Auto-adjust resolution', command=self.auto_adjust_resolution).grid(row=r, column=2, sticky='w')
        r += 1

        # style and blending
        ttk.Label(frm, text='Mosaic opacity (0 = mosaic only, 100 = original only):').grid(row=r, column=0, sticky='w')
        self.opacity_scale = ttk.Scale(frm, from_=0, to=100, orient='horizontal')
        self.opacity_scale.set(0)
        self.opacity_scale.grid(row=r, column=1, sticky='ew')
        r += 1

        # tile layout params (same as image app)
        ttk.Label(frm, text='Min tile size (px):').grid(row=r, column=0, sticky='w')
        self.min_tile = tk.IntVar(value=32)
        ttk.Spinbox(frm, from_=8, to=512, textvariable=self.min_tile).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Max tile size (px):').grid(row=r, column=0, sticky='w')
        self.max_tile = tk.IntVar(value=256)
        ttk.Spinbox(frm, from_=32, to=1024, textvariable=self.max_tile).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Variance threshold:').grid(row=r, column=0, sticky='w')
        self.var_threshold = tk.IntVar(value=650)
        ttk.Spinbox(frm, from_=10, to=5000, textvariable=self.var_threshold).grid(row=r, column=1, sticky='w')
        r += 1

        # stretch options
        ttk.Label(frm, text='Tile fit mode:').grid(row=r, column=0, sticky='w')
        self.stretch_mode = tk.StringVar(value='Keep aspect (crop)')
        self.stretch_combo = ttk.Combobox(frm, textvariable=self.stretch_mode, state='readonly', width=30)
        self.stretch_combo['values'] = ('Keep aspect (crop)', 'Scale to fit (letterbox)', 'Stretch (per-axis)')
        self.stretch_combo.grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Max stretch (% per axis, 100 = no stretch):').grid(row=r, column=0, sticky='w')
        self.max_stretch = tk.IntVar(value=300)
        ttk.Spinbox(frm, from_=100, to=1000, textvariable=self.max_stretch).grid(row=r, column=1, sticky='w')
        r += 1

        self.force_stretch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Force full stretch', variable=self.force_stretch_var).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1

        # merging and clustering
        self.merge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Enable merge groups (spanning tiles)', variable=self.merge_var).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1

        ttk.Label(frm, text='Merge coverage threshold (%):').grid(row=r, column=0, sticky='w')
        self.merge_thresh = tk.IntVar(value=85)
        ttk.Spinbox(frm, from_=10, to=100, textvariable=self.merge_thresh).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Hash hamming threshold:').grid(row=r, column=0, sticky='w')
        self.hash_thresh = tk.IntVar(value=6)
        ttk.Spinbox(frm, from_=0, to=64, textvariable=self.hash_thresh).grid(row=r, column=1, sticky='w')
        r += 1

        self.diag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Allow diagonal adjacency', variable=self.diag_var).grid(row=r, column=0, columnspan=2, sticky='w')
        r += 1

        ttk.Label(frm, text='Gap tolerance (px):').grid(row=r, column=0, sticky='w')
        self.gap_tol = tk.IntVar(value=2)
        ttk.Spinbox(frm, from_=0, to=50, textvariable=self.gap_tol).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Min tiles for merge:').grid(row=r, column=0, sticky='w')
        self.min_tiles_merge = tk.IntVar(value=3)
        ttk.Spinbox(frm, from_=1, to=200, textvariable=self.min_tiles_merge).grid(row=r, column=1, sticky='w')
        r += 1

        ttk.Label(frm, text='Min bbox area for merge:').grid(row=r, column=0, sticky='w')
        self.min_area_merge = tk.IntVar(value=5000)
        ttk.Spinbox(frm, from_=0, to=1000000, textvariable=self.min_area_merge).grid(row=r, column=1, sticky='w')
        r += 1

        # animation source mode
        ttk.Label(frm, text='Animated source behavior:').grid(row=r, column=0, sticky='w')
        self.anim_mode = tk.StringVar(value='cycle')
        self.anim_mode_combo = ttk.Combobox(frm, textvariable=self.anim_mode, state='readonly', values=('cycle','best'), width=20)
        self.anim_mode_combo.grid(row=r, column=1, sticky='w')
        ttk.Label(frm, text='(cycle = faster, best = slower but more color-faithful)').grid(row=r, column=2, sticky='w')
        r += 1

        # actions
        buttons = ttk.Frame(frm)
        buttons.grid(row=r, column=0, columnspan=3, pady=(6,6), sticky='ew')
        ttk.Button(buttons, text='Scan source folder (manual)', command=self.start_scan_sources).pack(side='left')
        ttk.Button(buttons, text='Preview (one frame)', command=self.start_preview).pack(side='left', padx=(6,0))
        ttk.Button(buttons, text='Generate Video/GIF Mosaic', command=self.start_generate).pack(side='left', padx=(6,0))
        ttk.Button(buttons, text='Stop', command=self.request_stop).pack(side='left', padx=(6,0))
        r += 1

        self.progress = ttk.Progressbar(frm, mode='determinate')
        self.progress.grid(row=r, column=0, columnspan=3, sticky='ew')
        r += 1
        self.status = ttk.Label(frm, text='Ready')
        self.status.grid(row=r, column=0, columnspan=3, sticky='w')
        r += 1

        # preview area and theme
        self.preview_label = ttk.Label(frm, text='Preview will appear here')
        self.preview_label.grid(row=r, column=0, columnspan=3, sticky='nsew')
        frm.rowconfigure(r, weight=1)
        r += 1

        self.dark_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text='Dark mode', variable=self.dark_var, command=self.apply_theme).grid(row=r, column=0, sticky='w')
        r += 1

        self.preview_imgtk = None
        self.apply_theme()

    # ---------- GUI helper methods ----------
    def set_status(self, txt):
        def _():
            self.status.config(text=txt)
        self.root.after(0, _)

    def update_progress(self, v, total):
        def _():
            self.progress['maximum'] = total
            self.progress['value'] = v
            self.status.config(text=f'Progress: {v}/{total}')
        self.root.after(0, _)

    def browse_target(self):
        p = filedialog.askopenfilename(filetypes=[('Video/GIF', '*.mp4;*.mov;*.avi;*.webm;*.gif'), ('All files','*.*')])
        if p:
            self.target_entry.delete(0, tk.END)
            self.target_entry.insert(0, p)

    def browse_src_and_scan(self):
        p = filedialog.askdirectory()
        if p:
            self.src_entry.delete(0, tk.END)
            self.src_entry.insert(0, p)
            self.start_scan_sources()

    def browse_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, p)

    def start_scan_sources(self):
        folder = self.src_entry.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror('Error', 'Select a valid source folder first')
            return
        t = threading.Thread(target=self.scan_sources_thread, args=(folder,), daemon=True)
        t.start()

    def scan_sources_thread(self, folder):
        self.set_status('Scanning source folder (this may take a while for videos)...')
        def prog(i, n): self.update_progress(i, n)
        try:
            self.src_index = build_source_index(folder, thumb_for_avg=32, hash_size=8, progress_callback=prog, max_frames_per_source=6)
            self.set_status(f'Scan complete: {len(self.src_index)} sources')
        except Exception as e:
            self.set_status('Error scanning sources')
            messagebox.showerror('Error', str(e))

    def request_stop(self):
        self.stop = True
        self.set_status('Stop requested...')

    def apply_theme(self):
        dark = bool(self.dark_var.get())
        bg = '#222222' if dark else '#f0f0f0'
        fg = '#ddd' if dark else '#000'
        style = ttk.Style()
        try:
            style.configure('.', background=bg, foreground=fg)
            style.configure('TLabel', background=bg, foreground=fg)
            style.configure('TFrame', background=bg)
            style.configure('TEntry', fieldbackground=bg, background=bg, foreground=fg)
            style.configure('TCombobox', fieldbackground=bg, background=bg, foreground=fg)
            style.map('TCombobox', fieldbackground=[('readonly', bg)], foreground=[('readonly', fg)])
            self.root.configure(bg=bg)
            self.preview_label.configure(background=bg)
        except Exception:
            pass

    # ---------- Auto-adjust resolution ----------
    def auto_adjust_resolution(self):
        """Analyze the target reference frame and suggest an output scale percent so tiles are more readable.
        Strategy: compute median tile width from quadtree at current min/max tile settings, then pick a scale so median tile >= desired_tile_px.
        """
        tpath = self.target_entry.get().strip()
        if not tpath or not os.path.isfile(tpath):
            messagebox.showerror('Error', 'Select a valid target first')
            return
        try:
            # pick a reference frame (similar to preview logic)
            if is_gif_file(tpath):
                with Image.open(tpath) as im:
                    try:
                        im.seek(im.n_frames // 2)
                    except Exception:
                        im.seek(0)
                    ref_pil = im.convert('RGB')
                    ref_np = np.asarray(ref_pil)
            else:
                clip = VideoFileClip(tpath)
                t = clip.duration / 2.0
                ref_np = clip.get_frame(min(t, max(clip.duration - 1e-3, 0)))
                clip.close()

            # compute tiles at native size
            min_t = int(self.min_tile.get())
            max_t = int(self.max_tile.get())
            var_th = float(self.var_threshold.get())
            tiles = quadtree_tiles(ref_np.astype(np.float32), min_tile=min_t, max_tile=max_t, var_threshold=var_th)
            if len(tiles) == 0:
                messagebox.showinfo('Auto-adjust', 'Could not compute tiles for reference frame; leaving scale at 100%')
                return
            widths = [tw for (x,y,tw,th) in tiles]
            median_w = float(np.median(widths)) if len(widths) > 0 else float(min_t)

            # desired visual tile size — aim for at least 48 px or at least min_tile
            desired_tile_px = max(48, min_t)
            recommended_pct = int(math.ceil((desired_tile_px / max(1.0, median_w)) * 100.0))
            # clamp to reasonable ranges
            recommended_pct = max(100, min(recommended_pct, 800))

            self.out_scale.set(recommended_pct)
            self.set_status(f'Auto-adjust set output scale to {recommended_pct}% (median tile {median_w:.1f}px -> target {desired_tile_px}px)')
            messagebox.showinfo('Auto-adjust', f'Recommended scale: {recommended_pct}%\nMedian tile width: {median_w:.1f}px\nThis will increase output resolution to make tiles more readable.')
        except Exception as e:
            messagebox.showerror('Auto-adjust error', str(e))

    # ---------- preview and generate ----------
    def start_preview(self):
        tpath = self.target_entry.get().strip()
        if not tpath or not os.path.isfile(tpath):
            messagebox.showerror('Error', 'Select a valid target first')
            return
        if not self.src_index:
            messagebox.showinfo('Info', 'Source index is empty; scanning first...')
            self.start_scan_sources()
            return
        thread = threading.Thread(target=self.preview_thread, daemon=True)
        self.stop = False
        thread.start()

    def preview_thread(self):
        try:
            self.set_status('Loading target (preview frame)...')
            tpath = self.target_entry.get().strip()
            if is_gif_file(tpath):
                # use PIL to pick a frame (preserve GIF semantics)
                with Image.open(tpath) as im:
                    if self.ref_frame_combo.get() == 'first':
                        im.seek(0)
                    else:
                        try:
                            im.seek(im.n_frames // 2)
                        except Exception:
                            im.seek(0)
                    ref_pil = im.convert('RGB')
                    ref_np = np.asarray(ref_pil)
            else:
                clip = VideoFileClip(tpath)
                if self.ref_frame_combo.get() == 'first':
                    ref_np = clip.get_frame(0)
                else:
                    t = clip.duration / 2.0
                    ref_np = clip.get_frame(min(t, max(clip.duration - 1e-3, 0)))
                clip.close()

            np_ref = ref_np.astype(np.float32)
            tiles = quadtree_tiles(np_ref, min_tile=int(self.min_tile.get()), max_tile=int(self.max_tile.get()), var_threshold=float(self.var_threshold.get()))
            self.set_status(f'{len(tiles)} tiles (preview)')

            src_colors = np.array([s['avg'] for s in self.src_index], dtype=np.float32) if len(self.src_index) > 0 else np.zeros((0,3), dtype=np.float32)
            tile_avgs = []
            for (x,y,tw,th) in tiles:
                reg = np_ref[y:y+th, x:x+tw]
                tile_avgs.append(reg.mean(axis=(0,1)) if reg.size else np.array([0,0,0]))
            if src_colors.shape[0] == 0:
                match_idx_ref = [-1]*len(tiles)
            else:
                match_idx_ref = []
                for ta in tile_avgs:
                    diffs = src_colors - ta
                    dists = np.einsum('ij,ij->i', diffs, diffs)
                    match_idx_ref.append(int(np.argmin(dists)))

            groups, src_cluster = compute_groups_for_tiles(tiles, match_idx_ref, [s['ahash'] for s in self.src_index], src_colors, hash_threshold=int(self.hash_thresh.get()), allow_diagonal=bool(self.diag_var.get()), gap_tolerance=int(self.gap_tol.get()))

            pil_ref = Image.fromarray(ref_np.astype(np.uint8)).convert('RGBA')
            overlay = Image.new('RGBA', pil_ref.size, (0,0,0,0))
            draw = ImageDraw.Draw(overlay)
            for g in groups:
                xs = [tiles[i][0] for i in g]
                ys = [tiles[i][1] for i in g]
                rights = [tiles[i][0]+tiles[i][2] for i in g]
                bottoms = [tiles[i][1]+tiles[i][3] for i in g]
                min_x = min(xs); min_y = min(ys)
                bbox_w = max(rights)-min_x; bbox_h = max(bottoms)-min_y
                area_sum = sum(tiles[i][2]*tiles[i][3] for i in g)
                cov = area_sum / max(1, bbox_w*bbox_h)
                if cov >= float(self.merge_thresh.get())/100.0 and len(g) >= int(self.min_tiles_merge.get()):
                    color = tuple([random.randint(50,220) for _ in range(3)]) + (110,)
                    draw.rectangle([min_x, min_y, min_x+bbox_w-1, min_y+bbox_h-1], fill=color, outline=(255,255,255,200))
            combined = Image.alpha_composite(pil_ref.convert('RGBA'), overlay)
            combined.thumbnail((800,600), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(combined)
            def show():
                self.preview_label.config(image=imgtk, text='')
                self.preview_imgtk = imgtk
                self.set_status('Preview ready (groups overlay).')
            self.root.after(0, show)
        except Exception as e:
            self.set_status('Error in preview')
            messagebox.showerror('Error', str(e))

    def start_generate(self):
        tpath = self.target_entry.get().strip()
        sfolder = self.src_entry.get().strip()
        outfolder = self.out_entry.get().strip()
        if not tpath or not os.path.isfile(tpath):
            messagebox.showerror('Error', 'Select a valid target.')
            return
        if not sfolder or not os.path.isdir(sfolder):
            messagebox.showerror('Error', 'Select a valid source folder.')
            return
        if not outfolder or not os.path.isdir(outfolder):
            messagebox.showerror('Error', 'Select a valid output folder.')
            return
        if not self.src_index:
            self.set_status('Source index empty -> scanning now...')
            self.start_scan_sources()
            return
        args = (
            tpath, outfolder,
            int(self.min_tile.get()), int(self.max_tile.get()), float(self.var_threshold.get()),
            bool(self.consistent_tiles_var.get()), int(self.frame_step.get()), int(self.max_frames.get()), int(self.out_fps.get()),
            float(self.opacity_scale.get())/100.0,
            self.stretch_combo.get(), float(self.max_stretch.get())/100.0, bool(self.force_stretch_var.get()),
            bool(self.merge_var.get()), float(self.merge_thresh.get())/100.0, int(self.hash_thresh.get()), bool(self.diag_var.get()), int(self.gap_tol.get()), int(self.min_tiles_merge.get()), int(self.min_area_merge.get()),
            self.anim_mode.get(), int(self.out_scale.get())
        )
        thread = threading.Thread(target=self.generate_thread, args=args, daemon=True)
        self.stop = False
        thread.start()

    def generate_thread(self, target_path, outfolder, min_t, max_t, var_th, consistent_tiles, frame_step, max_frames_limit, out_fps, opacity, stretch_combo, max_stretch_factor_pct, force_full, merge_enabled, merge_thresh, hash_thresh, allow_diag, gap_tol, min_tiles_merge, min_area_merge, anim_mode, out_scale_percent):
        try:
            self.set_status('Loading target...')
            target_ext = os.path.splitext(target_path)[1].lower()

            # read frames into memory depending on type (GIF preservation uses PIL)
            frames_np = []
            frame_durations_ms = None
            if is_gif_file(target_path):
                pil_frames = []
                durations_ms = []
                with Image.open(target_path) as im:
                    i = 0
                    try:
                        while True:
                            im.seek(i)
                            f = im.convert('RGB').copy()
                            pil_frames.append(f)
                            dur = im.info.get('duration', None)
                            if dur is None:
                                dur = int(1000 / max(1, out_fps))
                            durations_ms.append(int(dur))
                            i += 1
                    except EOFError:
                        pass
                # apply frame step and max_frames_limit
                indices = list(range(0, len(pil_frames), frame_step))
                if max_frames_limit > 0:
                    indices = indices[:max_frames_limit]
                frames_np = [np.asarray(pil_frames[i]) for i in indices]
                frame_durations_ms = [durations_ms[i] for i in indices]
                self.set_status(f'Extracted {len(frames_np)} GIF frames')
            else:
                clip = VideoFileClip(target_path)
                total_frames = int(math.ceil(clip.duration * clip.fps))
                indices = list(range(0, total_frames, frame_step))
                if max_frames_limit > 0:
                    indices = indices[:max_frames_limit]
                times = [min((i / clip.fps), clip.duration - 1e-3) for i in indices]
                self.set_status(f'Extracting {len(times)} frames (this may take a while)...')
                frames_np = []
                for t in tqdm(times):
                    if self.stop:
                        clip.close()
                        return
                    frames_np.append(clip.get_frame(t))
                clip.close()

            if len(frames_np) == 0:
                self.set_status('No frames to process')
                return

            # --- Apply output scaling (resample frames before tiling/rendering) ---
            scale_pct = max(10, int(out_scale_percent))
            scale = float(scale_pct) / 100.0
            if scale != 1.0:
                # rescale frames (this increases memory usage!)
                new_frames = []
                for f_np in frames_np:
                    pil = pil_from_frame(f_np.astype(np.uint8))
                    nw = max(1, int(round(pil.width * scale)))
                    nh = max(1, int(round(pil.height * scale)))
                    pil2 = pil.resize((nw, nh), Image.Resampling.LANCZOS)
                    new_frames.append(np.asarray(pil2))
                frames_np = new_frames
                self.set_status(f'Rescaled frames to {frames_np[0].shape[1]}x{frames_np[0].shape[0]} ({scale_pct}%)')

            # determine tiles/groups depending on consistent_tiles
            if consistent_tiles:
                mid = len(frames_np)//2
                ref_np = frames_np[mid].astype(np.float32)
                tiles = quadtree_tiles(ref_np, min_tile=min_t, max_tile=max_t, var_threshold=var_th)
                self.set_status(f'{len(tiles)} tiles computed (consistent)')
                # compute reference mappings for grouping
                src_colors = np.array([s['avg'] for s in self.src_index], dtype=np.float32) if len(self.src_index) > 0 else np.zeros((0,3), dtype=np.float32)
                tile_avgs_ref = []
                for (x,y,tw,th) in tiles:
                    reg = ref_np[y:y+th, x:x+tw]
                    tile_avgs_ref.append(reg.mean(axis=(0,1)) if reg.size else np.array([0.0,0.0,0.0]))
                if src_colors.shape[0] == 0:
                    match_idx_ref = [-1]*len(tiles)
                else:
                    match_idx_ref = []
                    for ta in tile_avgs_ref:
                        diffs = src_colors - ta
                        dists = np.einsum('ij,ij->i', diffs, diffs)
                        match_idx_ref.append(int(np.argmin(dists)))
                groups = None
                src_cluster = None
                if merge_enabled:
                    groups, src_cluster = compute_groups_for_tiles(tiles, match_idx_ref, [s['ahash'] for s in self.src_index], src_colors, hash_threshold=hash_thresh, allow_diagonal=allow_diag, gap_tolerance=gap_tol)

                compute_tiles_per_frame = False
            else:
                # per-frame computation: let render_frames_for_video compute tiles/groups each frame
                tiles = None
                groups = None
                src_cluster = None
                compute_tiles_per_frame = True
                self.set_status('Tiles will be computed per-frame (slower)')

            # translate stretch mode code
            sm = 'keep'
            if stretch_combo == 'Keep aspect (crop)':
                sm = 'keep'
            elif stretch_combo == 'Scale to fit (letterbox)':
                sm = 'fit'
            else:
                sm = 'stretch'
            max_stretch_factor = max(1.0, float(max_stretch_factor_pct))

            # render
            self.set_status('Rendering frames (this may take quite a while)...')
            def prog(i, n): self.update_progress(i, n)
            out_pils = render_frames_for_video(frames_np, tiles=tiles, src_index=self.src_index, groups=groups, src_cluster=src_cluster,
                                               stretch_mode=sm, max_stretch_factor=max_stretch_factor, force_full_stretch=force_full,
                                               merge_area_threshold=merge_thresh, min_group_tiles=min_tiles_merge, min_merge_bbox_area=min_area_merge,
                                               animate_source_mode=anim_mode, opacity=opacity,
                                               progress_callback=prog, stop_flag=lambda: self.stop,
                                               compute_tiles_per_frame=compute_tiles_per_frame, min_tile=min_t, max_tile=max_t, var_threshold=var_th,
                                               hash_threshold=hash_thresh, allow_diagonal=allow_diag, gap_tolerance=gap_tol)

            # Save output using matching container (GIF -> GIF, MP4 -> MP4). Default mp4 otherwise.
            ts = int(time.time())
            if is_gif_file(target_path):
                out_name = f'video_mosaic_{ts}.gif'
                outpath = os.path.join(outfolder, out_name)
                # durations
                durations = frame_durations_ms if frame_durations_ms is not None else [int(1000/out_fps)] * len(out_pils)
                try:
                    out_pils[0].save(outpath, save_all=True, append_images=out_pils[1:], duration=durations, loop=0)
                    self.set_status(f'Done. Saved to {outpath}')
                except Exception as e:
                    self.set_status('Error saving GIF')
                    messagebox.showerror('Error', str(e))
            else:
                out_name = f'video_mosaic_{ts}.mp4'
                outpath = os.path.join(outfolder, out_name)
                try:
                    seq = [np.asarray(im.convert('RGB')) for im in out_pils]
                    clip_out = ImageSequenceClip(seq, fps=out_fps)
                    clip_out.write_videofile(outpath, codec='libx264', audio=False, threads=4, verbose=False, logger=None)
                    clip_out.close()
                    self.set_status(f'Done. Saved to {outpath}')
                except Exception as e:
                    self.set_status('Error saving MP4')
                    messagebox.showerror('Error', str(e))

            # show first frame in preview
            preview = out_pils[0].copy()
            preview.thumbnail((800,600), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(preview)
            def show():
                self.preview_label.config(image=imgtk, text='')
                self.preview_imgtk = imgtk
            self.root.after(0, show)

        except Exception as ex:
            if str(ex) == 'Stopped by user':
                self.set_status('Generation stopped.')
            else:
                self.set_status('Error while generating')
                messagebox.showerror('Error', str(ex))


# ---------- run ----------
if __name__ == '__main__':
    root = tk.Tk()
    app = VideoMosaicApp(root)
    root.geometry('1100x900')
    root.mainloop()
