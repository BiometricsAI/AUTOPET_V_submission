"""
UAM_team_submission - AutoPET interactive router for Grand Challenge.

Usa el MISMO tracer_classifier.py externo que dice_score_inter_check.py (enrutado
identico) y hace inferencia in-process con 5 canales [CT, PET, prev_mask, dist_fg, dist_bg].

PERSISTENCIA ENTRE ITERACIONES: el contenedor es llamado UNA vez por iteracion
(los organizadores gestionan el bucle). Segun su correo, los datos que deban
sobrevivir entre iteraciones se escriben/leen en /output (NO /cache, que no
persiste en la plataforma). Por eso prev_mask se guarda en /output.

Fixes: clicks como voxel-index [i,j,k]->(z,y,x); transpose_forward en el mapeo;
guia == trainer._add_bump (exp, pico 1, sigma 10 FDG / 5 PSMA).
"""
import os
import sys
import json
import joblib
import importlib.util
import hashlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

TRAINERS_DIR = "/opt/algorithm/custom_trainers"
sys.path.insert(0, TRAINERS_DIR)

NNUNET_RESULTS = "/opt/algorithm/nnUNet_results"
# Persistencia entre iteraciones: /output (lo indican los organizadores). /cache NO persiste.
STATE_DIR = "/output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_NAME = "checkpoint_best.pth"

TR_CLASSIFIER_PY = "/opt/algorithm/tracer_classifier.py"
RF_MODEL_PATH = "/opt/algorithm/rf_tracer_classifier_ds003.joblib"
PSMA_CONF_THRESHOLD = 0.65


def _load_py(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class AutoPETRouter:
    def __init__(self):
        self.input_path = "/input/"
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        print("Loading Random Forest classifier (external module)...")
        bundle = joblib.load(RF_MODEL_PATH)
        self.rf_model = bundle["model"]
        self.rf_features = bundle["features"]
        self._idx_psma = list(self.rf_model.classes_).index(1)
        self._clf = _load_py(TR_CLASSIFIER_PY, "tracer_classifier_mod")

    # ---- mismo clasificador que dice_score_inter_check.build_classifier ----
    def classify_tracer(self, pet_path):
        try:
            clf = self._clf
            img_gray = clf.generar_mip_coronal_normalizado(pet_path)
            mask = clf.binarizar_y_limpiar_mascara(img_gray)
            zonas = clf.segmentar_franjas_guillotina(img_gray)
            if not zonas:
                return None
            masas = clf.extraer_masas_regionales(img_gray, mask, zonas)
            feats = clf.extraer_vector_caracteristicas(masas, img_gray, zonas)
            X = pd.DataFrame([feats]).reindex(columns=self.rf_features, fill_value=0.0)
            return float(self.rf_model.predict_proba(X)[0][self._idx_psma])
        except Exception as e:
            print(f"Tracer classify failed ({e}); defaulting to FDG route")
            return None

    def load_interactive_network(self, trainer_name, dataset_name, fold,
                                 plans_manager, config_manager, dataset_json):
        mod = _load_py(os.path.join(TRAINERS_DIR, f"{trainer_name}.py"), "trainer_mod")
        TrainerCls = getattr(mod, trainer_name)
        label_manager = plans_manager.get_label_manager(dataset_json)
        net = TrainerCls.build_network_architecture(
            plans_manager, config_manager, TrainerCls.N_IMAGE,
            label_manager.num_segmentation_heads, enable_deep_supervision=False)
        ckpt_path = os.path.join(NNUNET_RESULTS, dataset_name,
                                 f"{trainer_name}__nnUNetPlans__3d_fullres",
                                 f"fold_{fold}", CKPT_NAME)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
              for k, v in ckpt["network_weights"].items()}
        net.load_state_dict(sd)
        net.decoder.deep_supervision = False
        net.deep_supervision = False
        net.eval().to(DEVICE)
        print(f"Network loaded ({trainer_name}, epoch {ckpt.get('current_epoch', '?')})")
        return net, TrainerCls.N_IMAGE

    def add_bump(self, heat, coords, sigma):
        if len(coords) == 0:
            return
        D, H, W = heat.shape
        m = int(np.ceil(3.0 * sigma))
        for c in coords:
            z0, z1 = max(c[0]-m, 0), min(c[0]+m+1, D)
            y0, y1 = max(c[1]-m, 0), min(c[1]+m+1, H)
            x0, x1 = max(c[2]-m, 0), min(c[2]+m+1, W)
            zz = np.arange(z0, z1)[:, None, None]
            yy = np.arange(y0, y1)[None, :, None]
            xx = np.arange(x0, x1)[None, None, :]
            d2 = (zz-c[0])**2 + (yy-c[1])**2 + (xx-c[2])**2
            bump = np.exp(-d2 / (2.0 * sigma**2)).astype(np.float32)
            heat[z0:z1, y0:y1, x0:x1] = np.maximum(heat[z0:z1, y0:y1, x0:x1], bump)

    @staticmethod
    def map_point_orig_to_prep(idx_zyx, props, prep_shape, transpose_forward):
        p = [idx_zyx[transpose_forward[0]], idx_zyx[transpose_forward[1]], idx_zyx[transpose_forward[2]]]
        bbox = props['bbox_used_for_cropping']
        shp_crop = props['shape_after_cropping_and_before_resampling']
        out = []
        for i in range(3):
            ci = p[i] - bbox[i][0]
            if shp_crop[i] > 1:
                ci = ci * (prep_shape[i] / shp_crop[i])
            j = int(round(ci))
            out.append(min(max(j, 0), prep_shape[i] - 1))
        return tuple(out)

    def process(self):
        print(f"Device: {DEVICE}"
              + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

        # 1. Read inputs (MHA) and write CT/PET as NIfTI
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]
        img_ct = sitk.ReadImage(os.path.join(self.input_path, "images/ct/", ct_mha))
        img_pet = sitk.ReadImage(os.path.join(self.input_path, "images/pet/", pet_mha))

        os.makedirs("/tmp/case", exist_ok=True)
        f_ct, f_pet = "/tmp/case/case_0000.nii.gz", "/tmp/case/case_0001.nii.gz"
        sitk.WriteImage(img_ct, f_ct, True)
        sitk.WriteImage(img_pet, f_pet, True)

        # 2. Classify tracer (external classifier on the PET file)
        p_psma = self.classify_tracer(f_pet)
        route = "psma" if (p_psma is not None and p_psma >= PSMA_CONF_THRESHOLD) else "fdg"
        pp = f"{p_psma:.3f}" if p_psma is not None else "None"
        print(f"Tracer: P(PSMA)={pp}  ->  route={route.upper()}")

        # 3. Routing config
        if route == "psma":
            dataset_name, trainer_name, fold, sigma = "Dataset005_psma", "nnUNet_Interactive_sigma_5", 3, 5.0
        else:
            dataset_name, trainer_name, fold, sigma = "Dataset003_complete", "nnUNet_Interactive_sigma_10", 2, 10.0

        model_dir = os.path.join(NNUNET_RESULTS, dataset_name,
                                 f"{trainer_name}__nnUNetPlans__3d_fullres")
        plans = json.load(open(os.path.join(model_dir, "plans.json")))
        dataset_json = json.load(open(os.path.join(model_dir, "dataset.json")))
        transpose_forward = plans.get("transpose_forward", [0, 1, 2])
        plans_manager = PlansManager(plans)
        config_manager = plans_manager.get_configuration("3d_fullres")
        label_manager = plans_manager.get_label_manager(dataset_json)

        network, n_image = self.load_interactive_network(
            trainer_name, dataset_name, fold, plans_manager, config_manager, dataset_json)

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=False, device=DEVICE,
                                    verbose=False, allow_tqdm=False)
        predictor.plans_manager = plans_manager
        predictor.configuration_manager = config_manager
        predictor.dataset_json = dataset_json
        predictor.label_manager = label_manager
        predictor.network = network
        predictor.allowed_mirroring_axes = None
        predictor.list_of_parameters = [network.state_dict()]

        # 4. Preprocess CT + PET
        preprocessor = DefaultPreprocessor(verbose=False)
        data, _, props = preprocessor.run_case([f_ct, f_pet], None, plans_manager,
                                               config_manager, dataset_json)
        data = torch.from_numpy(data).float()
        img_in = data[:n_image]
        D, H, W = img_in.shape[1:]

        # 5. prev_mask desde /output (persiste entre iteraciones; vacio en la iteracion 0)
        prev_mask = np.zeros((D, H, W), dtype=np.float32)
        geo = (tuple(img_pet.GetSize()),
               tuple(round(v, 3) for v in img_pet.GetOrigin()),
               tuple(round(v, 3) for v in img_pet.GetSpacing()))
        geo_key = hashlib.md5(str(geo).encode()).hexdigest()[:16]
        state_file = os.path.join(STATE_DIR, f"prevmask_{geo_key}.npy")
        try:
            if os.path.exists(state_file):
                cached = np.load(state_file)
                if cached.shape == (D, H, W):
                    prev_mask = cached.astype(np.float32)
                    print("Loaded prev_mask from /output (later interaction)")
        except Exception as e:
            print(f"prev_mask load skipped: {e}")

        # 6. Guidance from clicks
        dist_fg = np.zeros((D, H, W), dtype=np.float32)
        dist_bg = np.zeros((D, H, W), dtype=np.float32)
        json_file = os.path.join(self.input_path, "lesion-clicks.json")
        if os.path.exists(json_file):
            clicks = json.load(open(json_file))
            fg, bg = [], []
            for pnt in clicks.get("points", []):
                c = pnt["point"]
                idx_prep = self.map_point_orig_to_prep((c[2], c[1], c[0]), props, (D, H, W), transpose_forward)
                (fg if pnt["name"] == "tumor" else bg).append(idx_prep)
            self.add_bump(dist_fg, fg, sigma)
            self.add_bump(dist_bg, bg, sigma)
            print(f"Clicks: {len(fg)} tumor / {len(bg)} background")

        # 7. 5-channel input + sliding window
        net_in = torch.cat([img_in,
                            torch.from_numpy(prev_mask)[None],
                            torch.from_numpy(dist_fg)[None],
                            torch.from_numpy(dist_bg)[None]], dim=0)
        print("nnUNet inference starting (in-process)...")
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(net_in)

        # 8. Save soft fg-prob to /output as next interaction's prev_mask
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            fgp_prep = torch.softmax(logits.float(), dim=0)[1].cpu().numpy().astype(np.float32)
            np.save(state_file, fgp_prep)
        except Exception as e:
            print(f"prev_mask save skipped: {e}")

        # 9. Back to ORIGINAL space + write .mha
        pred = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits.cpu().numpy(), plans_manager, config_manager, label_manager, props,
            return_probabilities=False)
        pred = np.asarray(pred).astype(np.uint8)
        out_img = sitk.GetImageFromArray(pred)
        out_img.CopyInformation(img_pet)
        out_mha = os.path.join(self.output_path, f"{uuid}.mha")
        sitk.WriteImage(out_img, out_mha, True)
        print(f"Completed! Output saved in {out_mha}")


if __name__ == "__main__":
    torch.set_num_threads(8)
    AutoPETRouter().process()