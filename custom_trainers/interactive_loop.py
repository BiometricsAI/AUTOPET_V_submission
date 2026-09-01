"""
Bucle interactivo adaptado para probar TU contenedor en local, replicando lo que
hace Grand Challenge: iteracion 0 sin clicks, luego acumula scribbles derivados del
error (GT) y vuelve a llamar al contenedor. Debe reproducir la curva de
dice_score_inter_check.py (iteracion 0 == su paso s1).

Cambios vs el original de los organizadores:
  - sin check_weights.sh, sin metrics.py (DMM inline), sin el 'continue' de FDG.
  - llama a TU test.sh (TEST_SH) que monta test/input|output|cache en el contenedor.

Layout esperado:
  INPUT_CASES/
      images/  ->  <caso>_0000.nii.gz (CT), <caso>_0001.nii.gz (PET)
      labels/  ->  <caso>.nii.gz (GT)
  INPUT_INTERFACE/   (== la carpeta que monta test.sh: input/ output/ cache/)

Uso:
  python interactive_loop.py \
      --input_cases   /ruta/cases \
      --input_interface /ruta/al/repo/test \
      --test_sh       /ruta/al/repo/test.sh \
      --result_dir    /ruta/resultados \
      --strategy      centerline --max_iters 6
"""
from pathlib import Path
import os, sys, json, argparse, shutil, subprocess, traceback, logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import label as cc_label

# simulate_scribbles.py debe estar junto a este script (o en el PYTHONPATH)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_scribbles import (
    simulate_scribble_from_label,
    scribbles_to_gc_format,
    gc_to_swfastedit_format,
)

sitk.ProcessObject_SetGlobalWarningDisplay(False)
_STRUCT = np.ones((3, 3, 3))


def setup_logger(log_file):
    lg = logging.getLogger("interactive_segmentation")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    for h in (logging.FileHandler(log_file), logging.StreamHandler()):
        h.setFormatter(fmt); lg.addHandler(h)
    return lg


def dice_score(pred, gt):
    pred = (pred > 0).astype(np.uint8); gt = (gt > 0).astype(np.uint8)
    inter = np.sum(pred * gt); denom = np.sum(pred) + np.sum(gt)
    return 1.0 if denom == 0 else 2.0 * inter / denom


def detection_f1(pred, gt):
    """DMM: F1 de deteccion por componente (26-conn), == compute_obj_detection_metrics."""
    a = (gt > 0).astype(np.uint8); b = (pred > 0).astype(np.uint8)
    ls_ref, n_ref = cc_label(a, structure=_STRUCT)
    ls_pred, n_pred = cc_label(b, structure=_STRUCT)
    tp = sum(1 for i in range(1, n_ref + 1) if np.any(b[ls_ref == i]))
    fn = n_ref - tp
    fp = sum(1 for i in range(1, n_pred + 1) if not np.any(a[ls_pred == i]))
    return 1.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)


def convert_mha_to_nii(mha, nii):
    sitk.WriteImage(sitk.ReadImage(mha), nii, True)


def clean_dir(d):
    if os.path.exists(d):
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_cases", required=True)
    ap.add_argument("--input_interface", required=True, help="carpeta que monta test.sh (input/ output/ cache/)")
    ap.add_argument("--test_sh", required=True, help="ruta a tu test.sh")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--strategy", default="centerline", choices=["centerline", "random", "boundary"])
    ap.add_argument("--max_iters", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.result_dir, "run.log"))

    image_dir = os.path.join(args.input_cases, "images")
    label_dir = os.path.join(args.input_cases, "labels")
    cts = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if "_0000" in f)
    pets = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if "_0001" in f)
    labels = sorted(os.path.join(label_dir, f) for f in os.listdir(label_dir))

    ct_dir = os.path.join(args.input_interface, "input", "images", "ct")
    pet_dir = os.path.join(args.input_interface, "input", "images", "pet")
    seg_dir = os.path.join(args.input_interface, "output", "images", "tumor-lesion-segmentation")
    cache_dir = os.path.join(args.input_interface, "cache")
    clicks_json = os.path.join(args.input_interface, "input", "lesion-clicks.json")
    for d in (ct_dir, pet_dir, seg_dir, cache_dir, os.path.dirname(clicks_json)):
        os.makedirs(d, exist_ok=True)

    out_scores = os.path.join(args.result_dir, "metric_scores.json")
    case_dict = {}

    for ct, pet, label in zip(cts, pets, labels):
        tag = os.path.basename(ct).replace(".nii.gz", "")
        logger.info(f"Processing case: {tag}")
        case_dict[tag] = []

        for d in (ct_dir, pet_dir, seg_dir):
            clean_dir(d)
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)   # persiste ENTRE iteraciones del caso (prev_mask)

        try:
            sitk.WriteImage(sitk.ReadImage(ct), os.path.join(ct_dir, f"case_{tag}.mha"))
            sitk.WriteImage(sitk.ReadImage(pet), os.path.join(pet_dir, f"case_{tag}.mha"))
            gt = nib.load(label).get_fdata()
            empty_gt = np.sum(gt) == 0
            prev_dice = prev_dmm = None

            for it in range(args.max_iters):
                logger.info(f"[{tag}] Iteration {it}")
                try:
                    if it == 0:
                        data = {"tumor": [], "background": []}
                    else:
                        if empty_gt:
                            dice = prev_dice or 0.0
                            dmm = prev_dmm or 0.0
                        seg_path = os.path.join(seg_dir, f"case_{tag}.mha")
                        seg_nii = seg_path.replace(".mha", ".nii.gz")
                        if not os.path.exists(seg_path):
                            raise FileNotFoundError("Missing segmentation")
                        convert_mha_to_nii(seg_path, seg_nii)
                        pred = nib.load(seg_nii).get_fdata()
                        os.remove(seg_nii)
                        with open(clicks_json) as f:
                            data = gc_to_swfastedit_format(json.load(f))
                        if pred.shape != gt.shape:
                            raise ValueError(f"Shape mismatch pred{pred.shape} vs gt{gt.shape}")
                        overseg = (pred == 1) & (gt == 0)
                        underseg = (pred == 0) & (gt == 1)
                        scr_bg, _, fp = simulate_scribble_from_label(overseg, args.strategy)
                        scr_fg, _, fn = simulate_scribble_from_label(underseg, args.strategy)
                        if fp <= fn:
                            data["tumor"] += scr_fg
                        else:
                            data["background"] += scr_bg

                    with open(clicks_json, "w") as f:
                        json.dump(scribbles_to_gc_format(data), f)

                    # ---- llamada a TU contenedor ----
                    subprocess.run(["bash", args.test_sh], timeout=1200, check=True)

                    seg_path = os.path.join(seg_dir, f"case_{tag}.mha")
                    seg_nii = seg_path.replace(".mha", ".nii.gz")
                    convert_mha_to_nii(seg_path, seg_nii)
                    pred = nib.load(seg_nii).get_fdata()
                    os.remove(seg_nii)

                    dice = dice_score(pred, gt)
                    dmm = detection_f1(pred, gt)
                except Exception as e:
                    logger.warning(f"[{tag}] Iteration {it} failed: {e}")
                    logger.debug(traceback.format_exc())
                    dice, dmm = 0.0, 0.0

                prev_dice, prev_dmm = float(dice), float(dmm)
                case_dict[tag].append({"iteration": it, "dice": float(dice), "dmm": float(dmm)})
                logger.info(f"[{tag}] Dice@{it}: {dice:.4f}  DMM@{it}: {dmm:.4f}")

        except Exception as e:
            logger.error(f"[{tag}] Case failed: {e}")
            logger.debug(traceback.format_exc())
            case_dict[tag] = [{"iteration": i, "dice": 0.0, "dmm": 0.0} for i in range(args.max_iters)]

        with open(out_scores, "w") as f:
            json.dump(case_dict, f, indent=4)

    # AUC (trapz sobre iteraciones)
    auc = {}
    for cid, recs in case_dict.items():
        recs = sorted(recs, key=lambda x: x["iteration"])
        its = np.array([r["iteration"] for r in recs], float)
        auc[cid] = {"auc_dice": float(np.trapz([r["dice"] for r in recs], its)),
                    "auc_dmm": float(np.trapz([r["dmm"] for r in recs], its))}
    with open(out_scores.replace(".json", "_AUC.json"), "w") as f:
        json.dump(auc, f, indent=4)
    logger.info("Done.")


if __name__ == "__main__":
    main()