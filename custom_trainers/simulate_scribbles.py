"""
Scribble Simulation

This script generates foreground and background scribbles from 3D binary masks (labels or model errors).
It supports multiple strategies (centerline, boundary, random) and produces Gaussian heatmaps as additional NIfTI files for interactive segmentation pipelines.

Outputs (nnUNet format):
- Foreground scribble heatmap (_0002.nii.gz)
- Background scribble heatmap (_0003.nii.gz)
"""

import os
import argparse
import random
import json
import cc3d

import numpy as np
import nibabel as nib
import networkx as nx

from skimage.morphology import skeletonize, dilation, disk, binary_dilation, ball
from skimage.segmentation import find_boundaries
from skimage.draw import line

from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

from pathlib import Path

# ------------------------------------------------
# SCRIBBLE METHODS
# ------------------------------------------------

def scribble_centerline(slice_mask, trunc_fraction=0.1):
    """
    Generate a centerline-based scribble from a binary slice mask.

    The method extracts a skeleton, finds the longest path between two endpoints,
    and returns a truncated centerline scribble.

    Args:
        slice_mask (np.ndarray): 2D binary mask of a slice.
        trunc_fraction (float): Fraction of endpoints to trim.

    Returns:
        tuple: (scribble, skeleton)
    """
    skeleton = skeletonize(slice_mask).astype(np.uint8)

    skel_cc = cc3d.connected_components(skeleton, connectivity=8)

    unique, counts = np.unique(skel_cc, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    counts_dict.pop(0, None)

    if len(counts_dict) == 0:
        return slice_mask.copy(), slice_mask.copy()

    largest = max(counts_dict, key=counts_dict.get)
    skeleton = (skel_cc == largest).astype(np.uint8)

    coords = np.argwhere(skeleton)

    if len(coords) < 2:
        return slice_mask.copy(), skeleton

    G = nx.Graph()

    for y, x in coords:
        G.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0] and
                    0 <= nx_ < skeleton.shape[1] and
                    skeleton[ny, nx_]):
                    G.add_edge((y, x), (ny, nx_))

    dist_matrix = cdist(coords, coords)
    idx = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)

    p1 = tuple(coords[idx[0]])
    p2 = tuple(coords[idx[1]])

    try:
        path = nx.shortest_path(G, source=p1, target=p2)
    except Exception:
        return slice_mask.copy(), skeleton

    path_coords = np.array(path)

    if len(path_coords) > 10:
        n = len(path_coords)
        start = int(n * trunc_fraction)
        end = int(n * (1 - trunc_fraction))
        path_coords = path_coords[start:end]

    scribble = np.zeros_like(slice_mask)

    for y, x in path_coords:
        if slice_mask[y, x]:
            scribble[y, x] = 1

    return scribble, skeleton


def scribble_random(slice_mask, seed=42):
    """
    Generate a random line scribble between two voxels.

    Args:
        slice_mask (np.ndarray): 2D binary mask.
        seed (int): Random seed.

    Returns:
        tuple: (scribble, None)
    """
    random.seed(seed)

    coords = np.argwhere(slice_mask)

    if len(coords) < 2:
        return slice_mask.copy(), None

    p1 = coords[random.randint(0, len(coords) - 1)]
    p2 = coords[random.randint(0, len(coords) - 1)]

    rr, cc = line(p1[0], p1[1], p2[0], p2[1])

    scribble = np.zeros_like(slice_mask)

    valid = (rr >= 0) & (rr < slice_mask.shape[0]) & (cc >= 0) & (cc < slice_mask.shape[1])

    rr, cc = rr[valid], cc[valid]

    for r, c in zip(rr, cc):
        if slice_mask[r, c]:
            scribble[r, c] = 1

    return scribble, None



def scribble_boundary(slice_mask, seed=42):
    """
    Generate a boundary-following scribble on the lesion edge.

    Args:
        slice_mask (np.ndarray): 2D binary mask.
        seed (int): Random seed.

    Returns:
        tuple: (scribble, boundary)
    """
    random.seed(seed)

    boundaries = find_boundaries(slice_mask, mode='inner').astype(np.uint8)

    coords = np.argwhere(boundaries)

    if len(coords) < 2:
        return slice_mask.copy(), None

    start = tuple(coords[random.randint(0, len(coords) - 1)])

    scribble = np.zeros_like(slice_mask)
    visited = set()
    current = start

    scribble[current] = 1
    visited.add(current)

    length = max(2, int(0.2 * len(coords)))

    for _ in range(length):
        y, x = current
        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx

                if (0 <= ny < boundaries.shape[0] and
                    0 <= nx_ < boundaries.shape[1] and
                    boundaries[ny, nx_] and
                    (ny, nx_) not in visited):
                    neighbors.append((ny, nx_))

        if not neighbors:
            break

        current = neighbors[0]
        scribble[current] = 1
        visited.add(current)

    return scribble, boundaries


# ------------------------------------------------
# HEATMAP GENERATION
# ------------------------------------------------

def generate_gaussian_heatmap(coords, shape, sigma=0):
    """
    Generate a Gaussian heatmap from sparse coordinates.

    Args:
        coords (list): List of [x, y, z] coordinates.
        shape (tuple): Output volume shape.
        sigma (float): Gaussian smoothing strength.

    Returns:
        np.ndarray: Smoothed heatmap volume.
    """
    heatmap = np.zeros(shape, dtype=np.float32)

    for coord in coords:
        if (0 <= coord[0] < shape[0] and
            0 <= coord[1] < shape[1] and
            0 <= coord[2] < shape[2]):
            heatmap[tuple(coord)] = 1.0

    return gaussian_filter(heatmap, sigma=sigma)


def save_heatmap_nifti(heatmap, reference_nifti, output_path):
    """
    Save a heatmap as a NIfTI file using a reference image for affine/header.

    Args:
        heatmap (np.ndarray): 3D volume.
        reference_nifti (str): Reference image path.
        output_path (str): Output file path.
    """
    ref = nib.load(reference_nifti)

    out = nib.Nifti1Image(heatmap.astype(np.float32), ref.affine, ref.header)
    nib.save(out, output_path)

    


# ------------------------------------------------
# SCRIBBLE PIPELINE
# ------------------------------------------------
def scribbles_to_gc_format(input_scribbles, gc_json_path=None):

    # check if the input is a path to a file
    if isinstance(input_scribbles, str) and os.path.exists(input_scribbles):
        original_json_path = Path(input_scribbles)
        with open(original_json_path, 'r') as f:
            json_data = json.load(f)
    else:
        json_data = input_scribbles

    fg_points = json_data.get('tumor', [])
    bg_points = json_data.get('background', [])
    gc_dict = {  
        "version": {"major": 1, "minor": 0},  
        "type": "Multiple points",  
        "points": []
    }
    for fg_point in fg_points:
        gc_dict['points'].append({'point': fg_point, 'name': 'tumor'})
    for bg_point in bg_points:
        gc_dict['points'].append({'point': bg_point, 'name': 'background'})

    # Save the GC format JSON
    if gc_json_path is not None:
        with open(gc_json_path, 'w') as f_gc:
            json.dump(gc_dict, f_gc)
        print(f'Finished converting to {gc_json_path} in the GC format!')
    else:
        return gc_dict

def gc_to_swfastedit_format(gc_dict):
    swfast_dict = {
        "tumor": [],
        "background": []
    }
    
    for point in gc_dict.get("points", []):
        if point["name"] == "tumor":
            swfast_dict["tumor"].append(point["point"])
        elif point["name"] == "background":
            swfast_dict["background"].append(point["point"])
    return swfast_dict

def get_random_k_components(label, k=5, seed=42):
    """
    Randomly sample connected components from a label volume.

    Args:
        label (np.ndarray): 3D binary label.
        k (int): Maximum number of components.
        seed (int): Random seed.

    Returns:
        tuple: (component_labels, selected_ids)
    """
    labels = cc3d.connected_components(label, connectivity=26)

    unique = np.unique(labels)
    unique = unique[unique != 0]

    if len(unique) == 0:
        return labels, []

    random.seed(seed)
    selected = random.sample(list(unique), min(k, len(unique)))

    return labels, selected


def generate_scribbles_for_components(labels, component_ids, strategy, seed):
    """
    Generate scribbles for multiple connected components.

    Args:
        labels (np.ndarray): Component label volume.
        component_ids (list): Selected component IDs.
        strategy (str): Scribble strategy.
        seed (int): Random seed.

    Returns:
        np.ndarray: Binary scribble volume.
    """
    scribble_vol = np.zeros_like(labels, dtype=np.uint8)

    for cid in component_ids:
        comp_mask = (labels == cid).astype(np.uint8)

        slice_sums = comp_mask.sum(axis=(0, 1))
        if slice_sums.max() == 0:
            continue

        best_slice = slice_sums.argmax()
        slice_mask = comp_mask[:, :, best_slice]

        try:
            if strategy == "centerline":
                scribble_slice, _ = scribble_centerline(slice_mask)
            elif strategy == "boundary":
                scribble_slice, _ = scribble_boundary(slice_mask, seed)
            else:
                scribble_slice, _ = scribble_random(slice_mask, seed)
        except Exception:
            scribble_slice, _ = scribble_random(slice_mask, seed)

        scribble_vol[:, :, best_slice] += scribble_slice

    return (scribble_vol > 0).astype(np.uint8)

def heatmap_from_coords(coords_xyz, shape, sigma=1.0):
    heatmap = np.zeros(shape, dtype=np.float32)
    for coord in coords_xyz:
        x, y, z = coord
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            heatmap[x, y, z] = 1.0
    # Smooth with Gaussian filter
    if sigma >= 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap


def generate_heatmap_from_scribbles(scribble_vol, sigma=0):
    """
    Convert scribble volume into Gaussian heatmap.

    Args:
        scribble_vol (np.ndarray): Binary scribble volume.
        sigma (float): Gaussian smoothing.

    Returns:
        np.ndarray: Heatmap volume.
    """
    coords = np.argwhere(scribble_vol > 0)

    if len(coords) == 0:
        return np.zeros_like(scribble_vol, dtype=np.float32)

    return generate_gaussian_heatmap(coords.tolist(), scribble_vol.shape, sigma=sigma)

def simulate_scribble_from_label(label_array, strategy="centerline", seed=42):
    """
    Slice-wise component selection:
    - iterate over slices
    - find largest 2D connected component per slice
    - pick slice with largest component overall
    """

    best_slice = None
    best_component = None
    best_area = 0

    # iterate over slices (z-axis)
    for z in range(label_array.shape[2]):
        slice_mask = label_array[:, :, z]

        if np.sum(slice_mask) == 0:
            continue

        # 2D connected components
        labels_2d = cc3d.connected_components(slice_mask, connectivity=8)

        unique, counts = np.unique(labels_2d, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        counts_dict.pop(0, None)

        if not counts_dict:
            continue

        # largest component in this slice
        largest_label = max(counts_dict, key=counts_dict.get)
        area = counts_dict[largest_label]

        if area > best_area:
            best_area = area
            best_slice = z
            best_component = (labels_2d == largest_label).astype(np.uint8)

    # fallback if nothing found
    if best_component is None:
        return [], False

    # generate scribble on best slice
    try:
        if strategy == "centerline":
            scribble_slice, _ = scribble_centerline(best_component)
        elif strategy == "boundary":
            scribble_slice, _ = scribble_boundary(best_component, seed)
        else:
            scribble_slice, _ = scribble_random(best_component, seed)
    except Exception:
        scribble_slice, _ = scribble_random(best_component, seed)

    # place into volume
    scribble_vol = np.zeros_like(label_array, dtype=np.uint8)
    scribble_vol[:, :, best_slice] = scribble_slice

    # convert to xyz format
    coords = np.argwhere(scribble_vol > 0)
    coords_xyz = [[int(c[0]), int(c[1]), int(c[2])] for c in coords]

    # determine class (foreground/background)
    label_cls = (np.sum(label_array * scribble_vol) > 0)

    size = np.sum(scribble_vol)

    return coords_xyz, label_cls, size


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--nifti", required=True, type=str) 
    parser.add_argument("--strategy", required=True, choices=["centerline","random","boundary"]) 
    parser.add_argument("--heatmap_out", type=str) 
    parser.add_argument("--seed", type=int, default=42) 
    args = parser.parse_args()

    img = nib.load(args.nifti)
    data = img.get_fdata().astype(np.uint8)

    os.makedirs(args.heatmap_out, exist_ok=True)

    # =========================
    # EMPTY LABEL 
    # =========================
    if np.sum(data) == 0:
        print("Empty label detected → writing empty scribble volumes")

        empty = np.zeros_like(data, dtype=np.float32)

        fg_out = os.path.join(
            args.heatmap_out,
            os.path.basename(args.nifti).replace('.nii.gz', '_0002.nii.gz')
        )
        bg_out = fg_out.replace('_0002.nii.gz', '_0003.nii.gz')

        save_heatmap_nifti(empty, args.nifti, fg_out)
        save_heatmap_nifti(empty, args.nifti, bg_out)

        print("FG heatmap saved (empty):", fg_out)
        print("BG heatmap saved (empty):", bg_out)

        # =========================
        # SAVE SCRIBBLE JSON + GC FORMAT
        # =========================


        coords_fg = []
        coords_bg = []

        scribble_json = {
            "tumor": coords_fg,
            "background": coords_bg
        }

        json_out = fg_out.replace('_0002.nii.gz', '_lesion-clicks.json')

        # save GC format
        scribbles_to_gc_format(scribble_json, json_out)

        print("Scribble JSON saved:", json_out)

    else:
        # =========================
        # FOREGROUND SCRIBBLE HEATMAP
        # =========================
        labels_fg, comp_ids_fg = get_random_k_components(data, k=5)

        scribble_fg = generate_scribbles_for_components(
            labels_fg, comp_ids_fg, args.strategy, args.seed
        )
        
        heatmap_fg = generate_heatmap_from_scribbles(scribble_fg, sigma=0)

        
        # =========================
        # BACKGROUND SCRIBBLE HEATMAP
        # =========================
        dilated = binary_dilation(data, ball(1))
        dilated = binary_dilation(dilated, ball(1))

        bg_region = (dilated.astype(np.uint8) - data.astype(np.uint8)) > 0
        bg_region = bg_region.astype(np.uint8)
        
        labels_bg, comp_ids_bg = get_random_k_components(bg_region, k=5)
        
        scribble_bg = generate_scribbles_for_components(
            labels_bg, comp_ids_bg, args.strategy, args.seed
        )

        heatmap_bg = generate_heatmap_from_scribbles(scribble_bg, sigma=0)


        # =========================
        # SAVE OUTPUTS
        # =========================

        fg_out = os.path.join(args.heatmap_out, os.path.basename(args.nifti).replace('.nii.gz', '_0002.nii.gz'))  # _0002
        bg_out = fg_out.replace('_0002.nii.gz', '_0003.nii.gz')
        
        save_heatmap_nifti(heatmap_fg, args.nifti, fg_out)
        save_heatmap_nifti(heatmap_bg, args.nifti, bg_out)

        print("FG heatmap saved:", fg_out)
        print("BG heatmap saved:", bg_out)
        
        # =========================
        # SAVE SCRIBBLE JSON + GC FORMAT
        # =========================
        coords_fg = np.argwhere(scribble_fg > 0)
        coords_bg = np.argwhere(scribble_bg > 0)

        coords_fg = [[int(x), int(y), int(z)] for x, y, z in coords_fg]
        coords_bg = [[int(x), int(y), int(z)] for x, y, z in coords_bg]

        scribble_json = {
            "tumor": coords_fg,
            "background": coords_bg
        }

        json_out = fg_out.replace('_0002.nii.gz', '_lesion-clicks.json')

        # save GC format
        scribbles_to_gc_format(scribble_json, json_out)

        print("Scribble JSON saved:", json_out)
