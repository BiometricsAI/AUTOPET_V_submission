"""
AutoPET router para Grand Challenge.

Clasifica el trazador (FDG/PSMA) con un RandomForest, enruta al modelo nnUNet
correspondiente y hace la INFERENCIA IN-PROCESS (NO usa nnUNetv2_predict).

Motivo: los trainers nnUNetTr4_Interactive* usan una arquitectura custom de 5
canales (2 imagen: CT,PET + 3 guia: prev_mask,dist_fg,dist_bg inyectados via
guidance_proj). nnUNetv2_predict / perform_actual_validation alimentan solo los
canales del dataset (=2), asi que x[:, 2:] queda vacio y la red revienta con
"Non-empty 5D data tensor expected ... [1, 0, ...]". Aqui montamos el tensor de
5 canales a mano, exactamente como el script de evaluacion offline.

===================  COSAS A VERIFICAR (marcadas con  >>> VERIFICAR)  ===================
1) Los .py de los trainers custom deben estar en /opt/algorithm/custom_trainers
   (ver Dockerfile). El nnunetv2 de pip NO los trae.
2) checkpoint_best.pth vs checkpoint_final.pth: ajusta CKPT_NAME si hace falta.
3) sigma (dist_sigma) esta en VOXELS PREPROCESADOS y debe coincidir con el __init__
   del trainer (Interactive1=10.0, Interactive2=5.0 segun tu GUIDANCE_BY_TRAINER).
4) El mapeo click(fisico) -> voxel PREPROCESADO (map_point_orig_to_prep) es un
   anadido de este contenedor (tu script offline trabajaba ya en preprocesado).
   Haz un sanity check de que las gaussianas caen sobre la lesion.
=========================================================================================
"""
import os
import sys
import json
import joblib
import importlib.util
import numpy as np
import SimpleITK as sitk
import cv2
import torch
from scipy.signal import find_peaks

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

# >>> VERIFICAR (1): carpeta con los .py de los trainers custom dentro de la imagen
TRAINERS_DIR = "/opt/algorithm/custom_trainers"
sys.path.insert(0, TRAINERS_DIR)

NNUNET_RESULTS = "/opt/algorithm/nnUNet_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_NAME = "checkpoint_best.pth"   # >>> VERIFICAR (2)


class AutoPETRouter:
    def __init__(self):
        self.input_path = "/input/"
        self.output_path = "/output/images/tumor-lesion-segmentation/"
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        print("Loading Random Forest classifier...")
        self.rf_bundle = joblib.load("/opt/algorithm/rf_tracer_classifier_ds003.joblib")
        self.rf_model = self.rf_bundle['model']
        self.rf_features = self.rf_bundle['features']

    # =====================================================================
    # ============  CLASIFICADOR DE TRAZADOR (sin cambios)  ===============
    # =====================================================================
    def generate_normalized_coronal_mip(self, data):
        mip_coronal = np.max(data, axis=1).T
        img_gray_raw = np.flipud(mip_coronal)
        m_min, m_max = img_gray_raw.min(), img_gray_raw.max()
        if m_max - m_min > 0:
            return ((img_gray_raw - m_min) / (m_max - m_min) * 255).astype(np.uint8)
        return np.zeros_like(img_gray_raw, dtype=np.uint8)

    def binarize_and_clean_mask(self, img_gray):
        active_pixels = img_gray[img_gray > 5]
        robust_max = np.percentile(active_pixels, 99) if len(active_pixels) > 0 else 255
        dynamic_threshold = max(robust_max * 0.15, 20.0)
        _, base_mask = cv2.threshold(img_gray, dynamic_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, kernel)

    def segment_guillotine_slices(self, img_gray):
        raw_y_profile = np.sum(img_gray > 5, axis=1)
        real_body_rows = np.where(raw_y_profile > 0)[0]
        if len(real_body_rows) == 0:
            return None
        body_top, body_bottom = real_body_rows[0], real_body_rows[-1]
        body_height = body_bottom - body_top
        neck_limit = int(body_top + body_height * 0.22)
        top_profile = np.sum(img_gray, axis=1)[body_top:neck_limit]
        neck_cut = body_top + np.argmax(top_profile) + np.argmin(top_profile[np.argmax(top_profile):]) if len(top_profile) > 5 and np.argmax(top_profile) + 2 < len(top_profile) else body_top + int(body_height * 0.12)
        pelvis_start, pelvis_end = int(body_top + body_height * 0.55), int(body_top + body_height * 0.85)
        pelvis_range = np.sum(img_gray, axis=1)[pelvis_start:pelvis_end]
        pelvis_cut = pelvis_start + np.argmin(pelvis_range) if len(pelvis_range) > 0 else pelvis_start + int((pelvis_end - pelvis_start) / 2)
        return {'cabeza': (body_top, neck_cut), 'abdomen': (neck_cut, pelvis_cut), 'pelvis': (pelvis_cut, body_bottom)}

    def extract_regional_masses(self, img_gray, binary_mask, limit_zones):
        groups = {'cabeza': 0.0, 'abdomen': 0.0, 'pelvis': 0.0}
        for region, (y_in, y_out) in limit_zones.items():
            if region == 'cabeza':
                img_crop = img_gray[y_in:y_out, :]
                groups[region] = float(np.sum(img_crop * (img_crop > 5).astype(float)))
            else:
                seg_mask = binary_mask[y_in:y_out, :]
                if np.sum(seg_mask) > 0:
                    rows, cols = np.where(np.sum(seg_mask, axis=1) > 0)[0], np.where(np.sum(seg_mask, axis=0) > 0)[0]
                    m_rec = binary_mask[y_in+rows[0]:y_in+rows[-1], cols[0]:cols[-1]] / 255.0
                    capt = img_gray[y_in+rows[0]:y_in+rows[-1], cols[0]:cols[-1]] * m_rec
                    groups[region] = float(np.sum(capt))
                else:
                    raw_seg = img_gray[y_in:y_out, :]
                    if np.sum(raw_seg > 5) > 0:
                        rows, cols = np.where(np.sum(raw_seg > 5, axis=1) > 0)[0], np.where(np.sum(raw_seg > 5, axis=0) > 0)[0]
                        groups[region] = float(np.sum(raw_seg[rows[0]:rows[-1], cols[0]:cols[-1]]))
        return groups

    def extract_feature_vector(self, groups, img_gray, zones):
        features = {}
        m_cab, m_abd, m_pel = groups['cabeza'], groups['abdomen'], groups['pelvis']
        m_total = m_cab + m_abd + m_pel
        features['ratio_cabeza'] = m_cab / m_total if m_total > 0 else 0.0
        features['ratio_abdomen'] = m_abd / m_total if m_total > 0 else 0.0
        features['ratio_pelvis'] = m_pel / m_total if m_total > 0 else 0.0

        y_in_cab, y_out_cab = zones['cabeza']
        y_in_abd, y_out_abd = zones['abdomen']
        max_cab = np.max(img_gray[y_in_cab:y_out_cab, :]) if y_out_cab > y_in_cab else 0
        max_abd = np.max(img_gray[y_in_abd:y_out_abd, :]) if y_out_abd > y_in_abd else 0
        features['ratio_max_abd_cab'] = max_abd / max_cab if max_cab > 0 else 0.0

        x_profile = np.sum(img_gray[y_in_cab:y_out_cab, :], axis=0).astype(float)
        img_width = len(x_profile)
        threshold = np.max(x_profile) * 0.08
        total_mass = np.sum(x_profile)
        com_x = int(np.sum(np.arange(img_width) * x_profile) / total_mass) if total_mass > 0 else img_width // 2

        x_st, x_en = com_x, com_x
        while x_st > 0 and x_profile[x_st] > threshold: x_st -= 1
        while x_en < img_width - 1 and x_profile[x_en] > threshold: x_en += 1
        x_st, x_en = max(0, x_st - 12), min(img_width, x_en + 12)
        head_width = x_en - x_st

        if head_width < 15 or head_width > img_width * 0.70:
            aw = int(img_width * 0.35)
            x_st, x_en = max(0, com_x - aw // 2), min(img_width, com_x + aw // 2)
            head_width = x_en - x_st

        head_profile = x_profile[x_st:x_en]
        max_abs = np.max(head_profile) if len(head_profile) > 0 else 0
        features['max_abs_cabeza'] = max_abs

        if max_abs > 0:
            psmooth = cv2.GaussianBlur(head_profile.astype(float), (9, 1), 0).ravel()
            peaks, _ = find_peaks(psmooth, distance=max(5, int(head_width * 0.20)), prominence=max(10.0, max_abs * 0.15))
            features['fwhm_ratio'] = np.sum(x_profile[x_st:x_en] > (max_abs * 0.50)) / head_width
            features['fw85m_ratio'] = np.sum(x_profile[x_st:x_en] > (max_abs * 0.85)) / head_width
            if len(peaks) >= 2:
                ord_idx = peaks[np.argsort(psmooth[peaks])[-2:]]
                idx_L, idx_R = int(x_st + min(ord_idx)), int(x_st + max(ord_idx))
                val_L, val_R = int(x_profile[idx_L]), int(x_profile[idx_R])
                if val_L > (max_abs * 0.30) and val_R > (max_abs * 0.30):
                    valley_b = x_profile[idx_L+2: idx_R-1]
                    valley = int(x_profile[idx_L + 2 + np.argmin(valley_b)]) if len(valley_b) > 0 else min(val_L, val_R)
                    features.update({'num_picos_reales': 2, 'distancia_picos': (idx_R - idx_L) / head_width, 'profundidad_valle': valley / min(val_L, val_R) if min(val_L, val_R) > 0 else 1.0, 'asimetria_picos': abs(val_L - val_R) / max_abs})
                else:
                    features.update({'num_picos_reales': 1, 'distancia_picos': 0.0, 'profundidad_valle': 1.0, 'asimetria_picos': 1.0})
            else:
                features.update({'num_picos_reales': 1 if len(peaks) == 1 else 0, 'distancia_picos': 0.0, 'profundidad_valle': 1.0, 'asimetria_picos': 1.0})
        else:
            features.update({'fwhm_ratio': 0, 'fw85m_ratio': 0, 'num_picos_reales': 0, 'distancia_picos': 0, 'profundidad_valle': 1.0, 'asimetria_picos': 0})
        return features

    def predict_tracer(self, pet_array):
        print("Analyzing tracer (FDG vs PSMA)...")
        img_gray = self.generate_normalized_coronal_mip(pet_array)
        mask = self.binarize_and_clean_mask(img_gray)
        zones = self.segment_guillotine_slices(img_gray)
        if not zones:
            return "FDG", 1.0
        masses = self.extract_regional_masses(img_gray, mask, zones)
        feat_dict = self.extract_feature_vector(masses, img_gray, zones)
        vector = [feat_dict.get(c, 0.0) for c in self.rf_features]
        X = np.array(vector).reshape(1, -1)
        probs = self.rf_model.predict_proba(X)[0]
        pred_idx = self.rf_model.predict(X)[0]
        tracer = "PSMA" if pred_idx == 1 else "FDG"
        confidence = probs[1] if pred_idx == 1 else probs[0]
        return tracer, confidence

    # =====================================================================
    # ==========  INFERENCIA IN-PROCESS (portado de tu script)  ==========
    # =====================================================================
    def load_interactive_network(self, trainer_name, dataset_name, fold,
                                 plans_manager, config_manager, dataset_json):
        # Importa la clase del trainer custom por ruta (== tu import_trainer_class)
        trainer_py = os.path.join(TRAINERS_DIR, f"{trainer_name}.py")
        spec = importlib.util.spec_from_file_location("trainer_mod", trainer_py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["trainer_mod"] = mod
        spec.loader.exec_module(mod)
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
        print(f"Red custom cargada ({trainer_name}, epoca {ckpt.get('current_epoch', '?')})")
        return net, TrainerCls.N_IMAGE

    def add_bump(self, heat, coords, sigma):
        # == tu add_bump: gaussiana en el volumen completo, acumulando por maximo
        if len(coords) == 0:
            return
        D, H, W = heat.shape
        m = int(np.ceil(3.0 * sigma))
        for c in coords:
            z0, z1 = max(c[0] - m, 0), min(c[0] + m + 1, D)
            y0, y1 = max(c[1] - m, 0), min(c[1] + m + 1, H)
            x0, x1 = max(c[2] - m, 0), min(c[2] + m + 1, W)
            zz = np.arange(z0, z1)[:, None, None]
            yy = np.arange(y0, y1)[None, :, None]
            xx = np.arange(x0, x1)[None, None, :]
            d2 = (zz - c[0])**2 + (yy - c[1])**2 + (xx - c[2])**2
            bump = np.exp(-d2 / (2.0 * sigma**2)).astype(np.float32)
            heat[z0:z1, y0:y1, x0:x1] = np.maximum(heat[z0:z1, y0:y1, x0:x1], bump)

    @staticmethod
    def map_point_orig_to_prep(idx_zyx, props, prep_shape):
        """Mapea un voxel (z,y,x) del espacio ORIGINAL al PREPROCESADO aplicando el
        mismo crop + resample de nnUNet. Basta para el centro de una gaussiana.
        >>> VERIFICAR (4): sanity check de que los bumps caen sobre la lesion."""
        bbox = props['bbox_used_for_cropping']                        # [[z0,z1],[y0,y1],[x0,x1]]
        shp_crop = props['shape_after_cropping_and_before_resampling']  # (z,y,x)
        out = []
        for i in range(3):
            ci = idx_zyx[i] - bbox[i][0]
            if shp_crop[i] > 1:
                ci = ci * (prep_shape[i] / shp_crop[i])
            j = int(round(ci))
            out.append(min(max(j, 0), prep_shape[i] - 1))
        return tuple(out)

    def process(self):
        print("Checking GPU:", torch.cuda.is_available())

        # 1. Inputs
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]
        img_ct = sitk.ReadImage(os.path.join(self.input_path, "images/ct/", ct_mha))
        img_pet = sitk.ReadImage(os.path.join(self.input_path, "images/pet/", pet_mha))
        arr_pet = sitk.GetArrayFromImage(img_pet)

        # 2. Clasificar trazador
        tracer, conf = self.predict_tracer(arr_pet)
        print(f"Classifier result: {tracer} (Confidence: {conf:.2f})")

        # 3. Routing.  >>> VERIFICAR (3): sigma en VOXELS PREPROCESADOS == __init__ del trainer
        if tracer == "PSMA" and conf > 0.65:
            print("=> PSMA (nnUNetTr4_Interactive2, fold 3, sigma 5)")
            dataset_name, trainer_name, fold, sigma = "Dataset005_psma", "nnUNetTr4_Interactive2", 3, 5.0
        else:
            print("=> FDG (nnUNetTr4_Interactive1, fold 2, sigma 10)")
            dataset_name, trainer_name, fold, sigma = "Dataset003_complete", "nnUNetTr4_Interactive1", 2, 10.0

        # 4. plans.json + dataset.json (viven en la carpeta del modelo entrenado)
        model_dir = os.path.join(NNUNET_RESULTS, dataset_name,
                                 f"{trainer_name}__nnUNetPlans__3d_fullres")
        plans = json.load(open(os.path.join(model_dir, "plans.json")))
        dataset_json = json.load(open(os.path.join(model_dir, "dataset.json")))
        plans_manager = PlansManager(plans)
        config_manager = plans_manager.get_configuration("3d_fullres")
        label_manager = plans_manager.get_label_manager(dataset_json)

        network, n_image = self.load_interactive_network(
            trainer_name, dataset_name, fold, plans_manager, config_manager, dataset_json)

        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                    use_mirroring=False, perform_everything_on_device=True,
                                    device=DEVICE, verbose=False, allow_tqdm=False)
        predictor.plans_manager = plans_manager
        predictor.configuration_manager = config_manager
        predictor.dataset_json = dataset_json
        predictor.label_manager = label_manager
        predictor.network = network
        predictor.allowed_mirroring_axes = None
        predictor.list_of_parameters = [network.state_dict()]

        # 5. Preprocesado de CT+PET (los 2 canales que ve el encoder)
        os.makedirs("/tmp/case", exist_ok=True)
        f_ct = "/tmp/case/case_0000.nii.gz"
        f_pet = "/tmp/case/case_0001.nii.gz"
        sitk.WriteImage(img_ct, f_ct, True)
        sitk.WriteImage(img_pet, f_pet, True)

        preprocessor = DefaultPreprocessor(verbose=False)
        data, _, props = preprocessor.run_case([f_ct, f_pet], None, plans_manager,
                                               config_manager, dataset_json)
        data = torch.from_numpy(data).float()
        img_in = data[:n_image]                       # (2, D, H, W) preprocesado
        D, H, W = img_in.shape[1:]

        # 6. Canales de guia a resolucion PREPROCESADA, desde los clicks
        prev_mask = np.zeros((D, H, W), dtype=np.float32)
        dist_fg = np.zeros((D, H, W), dtype=np.float32)
        dist_bg = np.zeros((D, H, W), dtype=np.float32)

        json_file = os.path.join(self.input_path, "lesion-clicks.json")
        if os.path.exists(json_file):
            clicks = json.load(open(json_file))
            fg, bg = [], []
            for p in clicks.get("points", []):
                vx = img_pet.TransformPhysicalPointToIndex(p["point"])   # (x,y,z) espacio original
                idx_prep = self.map_point_orig_to_prep((vx[2], vx[1], vx[0]), props, (D, H, W))
                (fg if p["name"] == "tumor" else bg).append(idx_prep)
            self.add_bump(dist_fg, np.array(fg), sigma)
            self.add_bump(dist_bg, np.array(bg), sigma)

        # 7. Tensor de 5 canales (orden == training) y sliding window
        net_in = torch.cat([img_in,
                            torch.from_numpy(prev_mask)[None],
                            torch.from_numpy(dist_fg)[None],
                            torch.from_numpy(dist_bg)[None]], dim=0)     # (5, D, H, W)
        print("nnUNet inference starting (in-process)...")
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(net_in)

        # 8. Volver al espacio ORIGINAL y guardar .mha
        pred = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits.cpu().numpy(), plans_manager, config_manager, label_manager, props,
            return_probabilities=False)
        pred = np.asarray(pred).astype(np.uint8)

        out_img = sitk.GetImageFromArray(pred)
        out_img.CopyInformation(img_pet)              # la seg vive en la rejilla PET/CT
        out_mha = os.path.join(self.output_path, f"{uuid}.mha")
        sitk.WriteImage(out_img, out_mha, True)
        print(f"Completed! Output saved in {out_mha}")


if __name__ == "__main__":
    AutoPETRouter().process()