#!/usr/bin/env python3
"""
Clasificador de TRAZADOR (FDG vs PSMA) por Random Forest sobre el MIP coronal
del PET. Version adaptada a tu peticion, con evaluacion CRUZADA:

  FLUJO (fijo por defecto):
    1) ENTRENA con Dataset003_complete  (FDG + PSMA)   -> guarda los pesos (joblib)
    2) EVALUA  con Dataset006_deepPSMA                 -> matriz de confusion
    3) METRICA CLAVE: precision de la clase PSMA considerando SOLO las
       predicciones con confianza  P(PSMA) > umbral  (por defecto 0.65).

  Idea de la metrica: en tu pipeline, si un caso se clasifica como FDG (o como
  PSMA con confianza baja) se envia al MODELO CONJUNTO (fallback seguro). Si se
  clasifica como PSMA con confianza > 0.65 se envia a la RUTA PSMA especializada.
  El error costoso es un FDG etiquetado como PSMA con alta confianza. Por eso lo
  que se mide es: de todos los casos enviados a la ruta PSMA (pred=PSMA & conf>0.65),
  ¿que fraccion son realmente PSMA?  ==  PRECISION de PSMA @ umbral.

  NOTA: esto solo es informativo si DeepPSMA contiene AMBOS trazadores (fdg_* y
  psma_*), igual que asume tu script de DICE (detect_tracer). Si fuese PSMA puro,
  la precision saldria 100% trivialmente (no hay FDG con los que equivocarse).

RUTAS: se derivan de la MISMA convencion nnUNet que ya usas en el script de DICE
(tr_classifier.py):
    {NNUNET_RAW}/{DATASET}/imagesTr/{caso}_{canalPET}.nii.gz
La etiqueta real se deduce del prefijo del nombre:  fdg_* -> 0 (FDG),  psma_* -> 1 (PSMA).
El canal PET se autodetecta leyendo dataset.json (channel_names con 'PET'/'SUV');
si no se puede, se usa el sufijo por defecto (0001, como en tu version original).

Las funciones de EXTRACCION DE CARACTERISTICAS son COPIA VERBATIM de tu
tr_classifier_ml.py (no se ha tocado su logica).
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
import joblib
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# matplotlib es opcional (solo para guardar las matrices en PNG)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# =====================================================================
# 0. CONFIG POR DEFECTO  (rutas tomadas del script de DICE tr_classifier.py)
# =====================================================================
NNUNET_RAW     = "/home/pablolozano/AUTOPET_clean/nnunet_raw"   # == raiz de tu gt_folder
TRAIN_DATASET  = "Dataset003_complete"     # entrenamiento (FDG + PSMA)
TEST_DATASET   = "Dataset006_deepPSMA"     # evaluacion (DeepPSMA)  <-- VERIFICA el nombre EXACTO de la carpeta
CHANNEL_SUFFIX = "0001.nii.gz"             # canal PET por defecto (como en tu version original)
CONF_THRESHOLD = 0.65                      # umbral de confianza para "confiar" en la clase PSMA
MODEL_OUT      = "rf_tracer_classifier_ds003.joblib"
OUT_DIR        = "tracer_clf_crosseval_results"

TRACER_NAMES = {0: "FDG", 1: "PSMA"}       # 0 = FDG, 1 = PSMA (igual que tu obtener_ground_truth)


# =====================================================================
# 1. FUNCIONES BASE (CARGA Y SEGMENTACION)  ---  COPIA VERBATIM
# =====================================================================
def generar_mip_coronal_normalizado(path_nifti):
    img_nii = nib.load(path_nifti)
    data = img_nii.get_fdata()
    mip_coronal = np.max(data, axis=1).T
    img_gray_raw = np.flipud(mip_coronal)

    m_min, m_max = img_gray_raw.min(), img_gray_raw.max()
    if m_max - m_min > 0:
        return ((img_gray_raw - m_min) / (m_max - m_min) * 255).astype(np.uint8)
    return np.zeros_like(img_gray_raw, dtype=np.uint8)


def binarizar_y_limpiar_mascara(img_gray):
    pixeles_activos = img_gray[img_gray > 5]
    max_robusto = np.percentile(pixeles_activos, 99) if len(pixeles_activos) > 0 else 255
    umbral_dinamico = max(max_robusto * 0.15, 20.0)
    _, mascara_base = cv2.threshold(img_gray, umbral_dinamico, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_abierta = cv2.morphologyEx(mascara_base, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mascara_abierta, cv2.MORPH_CLOSE, kernel)


def segmentar_franjas_guillotina(img_gray):
    perfil_crudo_y = np.sum(img_gray > 5, axis=1)
    filas_cuerpo_real = np.where(perfil_crudo_y > 0)[0]
    if len(filas_cuerpo_real) == 0:
        return None
    body_top, body_bottom = filas_cuerpo_real[0], filas_cuerpo_real[-1]
    body_height = body_bottom - body_top

    limite_cuello = int(body_top + body_height * 0.22)
    perfil_sup = np.sum(img_gray, axis=1)[body_top:limite_cuello]
    corte_cuello = body_top + np.argmax(perfil_sup) + np.argmin(perfil_sup[np.argmax(perfil_sup):]) if len(perfil_sup) > 5 and np.argmax(perfil_sup) + 2 < len(perfil_sup) else body_top + int(body_height * 0.12)

    inicio_pel, fin_pel = int(body_top + body_height * 0.55), int(body_top + body_height * 0.85)
    rango_pelvis = np.sum(img_gray, axis=1)[inicio_pel:fin_pel]
    corte_pelvis = inicio_pel + np.argmin(rango_pelvis) if len(rango_pelvis) > 0 else inicio_pel + int((fin_pel - inicio_pel) / 2)

    return {'cabeza': (body_top, corte_cuello), 'abdomen': (corte_cuello, corte_pelvis), 'pelvis': (corte_pelvis, body_bottom)}


def extraer_masas_regionales(img_gray, mascara_binaria, zonas_limites):
    grupos_anatomicos = {'cabeza': 0.0, 'abdomen': 0.0, 'pelvis': 0.0}
    for region, (y_in, y_fin) in zonas_limites.items():
        if region == 'cabeza':
            img_crop = img_gray[y_in:y_fin, :]
            grupos_anatomicos[region] = float(np.sum(img_crop * (img_crop > 5).astype(float)))
        else:
            seg_masc = mascara_binaria[y_in:y_fin, :]
            if np.sum(seg_masc) > 0:
                filas, columnas = np.where(np.sum(seg_masc, axis=1) > 0)[0], np.where(np.sum(seg_masc, axis=0) > 0)[0]
                m_rec = mascara_binaria[y_in+filas[0]:y_in+filas[-1], columnas[0]:columnas[-1]] / 255.0
                capt = img_gray[y_in+filas[0]:y_in+filas[-1], columnas[0]:columnas[-1]] * m_rec
                grupos_anatomicos[region] = float(np.sum(capt))
            else:
                seg_crudo = img_gray[y_in:y_fin, :]
                if np.sum(seg_crudo > 5) > 0:
                    filas, columnas = np.where(np.sum(seg_crudo > 5, axis=1) > 0)[0], np.where(np.sum(seg_crudo > 5, axis=0) > 0)[0]
                    grupos_anatomicos[region] = float(np.sum(seg_crudo[filas[0]:filas[-1], columnas[0]:columnas[-1]]))
    return grupos_anatomicos


# =====================================================================
# 2. EXTRACCION DE CARACTERISTICAS PARA MACHINE LEARNING  ---  COPIA VERBATIM
# =====================================================================
def extraer_vector_caracteristicas(grupos_anatomicos, img_gray, zonas_limites):
    """Extrae caracteristicas numericas para el Random Forest."""
    features = {}

    # --- FEATURES GEOMETRICOS ---
    m_cab, m_abd, m_pel = grupos_anatomicos['cabeza'], grupos_anatomicos['abdomen'], grupos_anatomicos['pelvis']
    m_total = m_cab + m_abd + m_pel

    features['ratio_cabeza'] = m_cab / m_total if m_total > 0 else 0.0
    features['ratio_abdomen'] = m_abd / m_total if m_total > 0 else 0.0
    features['ratio_pelvis'] = m_pel / m_total if m_total > 0 else 0.0

    # --- FEATURE: CONTRASTE CABEZA VS ABDOMEN ---
    y_in_cab, y_fin_cab = zonas_limites['cabeza']
    y_in_abd, y_fin_abd = zonas_limites['abdomen']
    max_cab_crudo = np.max(img_gray[y_in_cab:y_fin_cab, :]) if y_fin_cab > y_in_cab else 0
    max_abd_crudo = np.max(img_gray[y_in_abd:y_fin_abd, :]) if y_fin_abd > y_in_abd else 0

    features['ratio_max_abd_cab'] = max_abd_crudo / max_cab_crudo if max_cab_crudo > 0 else 0.0

    # --- FEATURES TOPOLOGICOS (CABEZA) ---
    perfil_x = np.sum(img_gray[y_in_cab:y_fin_cab, :], axis=0).astype(float)
    ancho_img = len(perfil_x)

    umbral_ruido = np.max(perfil_x) * 0.08
    masa_total_x = np.sum(perfil_x)
    com_x = int(np.sum(np.arange(ancho_img) * perfil_x) / masa_total_x) if masa_total_x > 0 else ancho_img // 2

    x_st, x_en = com_x, com_x
    while x_st > 0 and perfil_x[x_st] > umbral_ruido:
        x_st -= 1
    while x_en < ancho_img - 1 and perfil_x[x_en] > umbral_ruido:
        x_en += 1
    x_st, x_en = max(0, x_st - 12), min(ancho_img, x_en + 12)

    ancho_cab = x_en - x_st
    if ancho_cab < 15 or ancho_cab > ancho_img * 0.70:
        aw = int(ancho_img * 0.35)
        x_st, x_en = max(0, com_x - aw // 2), min(ancho_img, com_x + aw // 2)
        ancho_cab = x_en - x_st

    perfil_cab = perfil_x[x_st:x_en]
    max_abs = np.max(perfil_cab) if len(perfil_cab) > 0 else 0
    features['max_abs_cabeza'] = max_abs

    if max_abs > 0:
        perfil_smooth = cv2.GaussianBlur(perfil_cab.astype(float), (9, 1), 0).ravel()
        picos_idx, _ = find_peaks(perfil_smooth, distance=max(5, int(ancho_cab * 0.20)), prominence=max(10.0, max_abs * 0.15))

        features['fwhm_ratio'] = np.sum(perfil_x[x_st:x_en] > (max_abs * 0.50)) / ancho_cab
        features['fw85m_ratio'] = np.sum(perfil_x[x_st:x_en] > (max_abs * 0.85)) / ancho_cab

        if len(picos_idx) >= 2:
            idx_ord = picos_idx[np.argsort(perfil_smooth[picos_idx])[-2:]]
            idx_L, idx_R = int(x_st + min(idx_ord)), int(x_st + max(idx_ord))
            val_L, val_R = int(perfil_x[idx_L]), int(perfil_x[idx_R])

            if val_L > (max_abs * 0.30) and val_R > (max_abs * 0.30):
                busqueda_valle = perfil_x[idx_L+2: idx_R-1]
                valle_central = int(perfil_x[idx_L + 2 + np.argmin(busqueda_valle)]) if len(busqueda_valle) > 0 else min(val_L, val_R)

                features['num_picos_reales'] = 2
                features['distancia_picos'] = (idx_R - idx_L) / ancho_cab
                features['profundidad_valle'] = valle_central / min(val_L, val_R) if min(val_L, val_R) > 0 else 1.0
                features['asimetria_picos'] = abs(val_L - val_R) / max_abs
            else:
                features['num_picos_reales'] = 1
                features['distancia_picos'] = 0.0
                features['profundidad_valle'] = 1.0
                features['asimetria_picos'] = 1.0
        else:
            features['num_picos_reales'] = 1 if len(picos_idx) == 1 else 0
            features['distancia_picos'] = 0.0
            features['profundidad_valle'] = 1.0
            features['asimetria_picos'] = 1.0
    else:
        features.update({
            'fwhm_ratio': 0, 'fw85m_ratio': 0, 'num_picos_reales': 0,
            'distancia_picos': 0, 'profundidad_valle': 1.0, 'asimetria_picos': 0
        })

    return features


# =====================================================================
# 3. UTILIDADES DE RUTAS Y ETIQUETAS
# =====================================================================
def images_dir(dataset):
    return os.path.join(NNUNET_RAW, dataset, "imagesTr")


def obtener_ground_truth(nombre_archivo):
    """0 = FDG, 1 = PSMA. Prefijo estricto (coherente con detect_tracer del script
    de DICE); con fallback a la 1a letra por compatibilidad con tu version original."""
    low = nombre_archivo.lower()
    if low.startswith('psma_'):
        return 1
    if low.startswith('fdg_'):
        return 0
    if low.startswith('p'):
        return 1
    if low.startswith('f'):
        return 0
    return None


def detectar_sufijo_pet(dataset, fallback=CHANNEL_SUFFIX):
    """Lee dataset.json y devuelve el sufijo del canal PET ('0001.nii.gz', etc.).
    Busca en channel_names / modality un nombre que contenga 'PET' o 'SUV'.
    Si no lo encuentra, usa el fallback (tu sufijo original)."""
    dj = os.path.join(NNUNET_RAW, dataset, "dataset.json")
    try:
        meta = json.load(open(dj))
        ch = meta.get("channel_names") or meta.get("modality") or {}
        for k, v in ch.items():
            if any(t in str(v).lower() for t in ("pet", "suv")):
                return f"{int(k):04d}.nii.gz"
    except Exception:
        pass
    return fallback


def obtener_archivos_pet(dataset, sufijo):
    """Lista los NIfTI del canal PET de un dataset (imagesTr)."""
    d = images_dir(dataset)
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, f"*_{sufijo}")))


# =====================================================================
# 4. EXTRACCION DE FEATURES DE UN DATASET
# =====================================================================
def extraer_features_dataset(lista_rutas, ruta_salida_csv, desc_tqdm="Procesando"):
    """Extrae el vector de features de cada PET y devuelve un DataFrame."""
    datos, n_fail = [], 0
    for path in tqdm(lista_rutas, desc=desc_tqdm, unit="img"):
        nombre = os.path.basename(path)
        label = obtener_ground_truth(nombre)
        if label is None:
            continue
        try:
            img_gray = generar_mip_coronal_normalizado(path)
            mascara = binarizar_y_limpiar_mascara(img_gray)
            zonas = segmentar_franjas_guillotina(img_gray)
            if zonas:
                masas = extraer_masas_regionales(img_gray, mascara, zonas)
                vector_features = extraer_vector_caracteristicas(masas, img_gray, zonas)
                vector_features['filename'] = nombre
                vector_features['true_label'] = label
                datos.append(vector_features)
            else:
                n_fail += 1
        except Exception:
            n_fail += 1   # no rompemos la barra; contamos los fallos

    df = pd.DataFrame(datos)
    if len(df) > 0:
        df.to_csv(ruta_salida_csv, index=False)
        dist = df['true_label'].map(TRACER_NAMES).value_counts().to_dict()
        print(f"[+] Features guardadas en: {ruta_salida_csv}  ({len(df)} casos, {dist})")
    else:
        print(f"[!] No se extrajeron features validas para {ruta_salida_csv}.")
    if n_fail:
        print(f"    ({n_fail} imagen(es) sin features validas / con error)")
    return df


def cargar_o_extraer(dataset, csv_path, sufijo, reuse):
    if reuse and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[=] Reusando features cacheadas de {csv_path} ({len(df)} casos)")
        return df
    rutas = obtener_archivos_pet(dataset, sufijo)
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron '*_{sufijo}' en {images_dir(dataset)}.\n"
            f"    Revisa NNUNET_RAW / nombre del dataset / sufijo del canal PET.")
    print(f"[i] {dataset}: {len(rutas)} PET(s) canal '{sufijo}' en {images_dir(dataset)}")
    return extraer_features_dataset(rutas, csv_path, desc_tqdm=f"Features {dataset}")


# =====================================================================
# 5. ENTRENAMIENTO
# =====================================================================
def entrenar(df_train, model_out):
    y = df_train['true_label'].values
    X = df_train.drop(columns=['filename', 'true_label'])
    feat_cols = list(X.columns)

    if len(np.unique(y)) < 2:
        raise ValueError("El conjunto de entrenamiento tiene una sola clase; "
                         "no se puede entrenar el clasificador FDG/PSMA.")

    print("\n=== ENTRENANDO RANDOM FOREST (Dataset de entrenamiento) ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(X, y)

    # Guardamos modelo + orden de features (imprescindible para inferencia coherente)
    joblib.dump({'model': rf, 'features': feat_cols,
                 'tracer_names': TRACER_NAMES}, model_out)
    print(f"[+] Pesos guardados en: {model_out}")

    # Importancia de variables (informativo)
    imp = sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\n--- IMPORTANCIA DE LAS VARIABLES ---")
    for f, v in imp:
        print(f"{f.ljust(22)}: {v*100:.1f}%")
    return rf, feat_cols


# =====================================================================
# 6. VISUALIZACION DE MATRICES
# =====================================================================
def print_matrix(mat, row_labels, col_labels, title):
    mat = np.asarray(mat).astype(int)
    cw = max(10, max(len(c) for c in col_labels) + 2)
    print("\n" + title)
    print(" " * 14 + "".join(f"{c:>{cw}}" for c in col_labels) + "   <- PREDICHO / RUTA")
    for i, r in enumerate(row_labels):
        print(f"{('real ' + r):>14}" + "".join(f"{mat[i, j]:>{cw}d}" for j in range(mat.shape[1])))


def save_matrix_png(mat, row_labels, col_labels, title, path, xlabel="Predicho"):
    if not _HAS_MPL:
        return
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Real")
    ax.set_title(title)
    thr = mat.max() / 2 if mat.max() > 0 else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha='center', va='center',
                    color='white' if mat[i, j] > thr else 'black', fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[+] Matriz guardada en: {path}")


# =====================================================================
# 7. EVALUACION CRUZADA + METRICA PSMA @ CONFIANZA
# =====================================================================
def evaluar(rf, feat_cols, df_test, conf_threshold, out_dir, tag=""):
    # Alinear columnas EXACTAMENTE al orden de entrenamiento
    faltan = [c for c in feat_cols if c not in df_test.columns]
    if faltan:
        print(f"[!] Aviso: faltan features en test {faltan} -> se rellenan con 0.")
    X = df_test.reindex(columns=feat_cols, fill_value=0.0)
    y_true = df_test['true_label'].values.astype(int)

    proba = rf.predict_proba(X)
    idx_psma = list(rf.classes_).index(1)          # posicion de la clase PSMA(=1)
    p_psma = proba[:, idx_psma]
    y_pred = rf.predict(X).astype(int)

    clases_test = np.unique(y_true)
    solo_una = len(clases_test) < 2

    # -------- 7.1 Metricas estandar (argmax) --------
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 70)
    print(f"EVALUACION CRUZADA {tag}")
    print("=" * 70)
    print(f"Accuracy global (argmax): {acc*100:.2f}%   |   casos: {len(y_true)}")
    if solo_una:
        print(f"[!] OJO: el test solo contiene la clase "
              f"{TRACER_NAMES[int(clases_test[0])]}. La 'precision PSMA' sera trivial.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])   # filas=real, col=pred (FDG,PSMA)
    print_matrix(cm, ["FDG", "PSMA"], ["FDG", "PSMA"], "MATRIZ DE CONFUSION (argmax)")
    print("\n--- REPORTE DE CLASIFICACION ---")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=['FDG (0)', 'PSMA (1)'], zero_division=0))

    # -------- 7.2 METRICA CLAVE: precision de PSMA con P(PSMA) > umbral --------
    # "clasificado como PSMA con confianza>umbral" (para umbral>=0.5 equivale a p_psma>umbral)
    mask_conf_psma = (y_pred == 1) & (p_psma > conf_threshold)
    n_conf = int(mask_conf_psma.sum())
    n_conf_ok = int(np.sum(mask_conf_psma & (y_true == 1)))   # realmente PSMA
    n_conf_err = n_conf - n_conf_ok                           # eran FDG (error costoso)
    prec_conf = (n_conf_ok / n_conf) if n_conf > 0 else float('nan')

    total_psma = int(np.sum(y_true == 1))
    cobertura = (n_conf_ok / total_psma) if total_psma > 0 else float('nan')

    print("\n" + "-" * 70)
    print(f"PSMA CON CONFIANZA  P(PSMA) > {conf_threshold:.2f}")
    print("-" * 70)
    print(f"  Casos enviados a la ruta PSMA (pred=PSMA & conf>{conf_threshold:.2f}) : {n_conf}")
    print(f"     de ellos REALMENTE PSMA (aciertos)                        : {n_conf_ok}")
    print(f"     de ellos REALMENTE FDG  (errores costosos)                : {n_conf_err}")
    if n_conf > 0:
        print(f"  >> PRECISION PSMA @ {conf_threshold:.2f}  = {prec_conf*100:.2f}%")
    else:
        print(f"  >> PRECISION PSMA @ {conf_threshold:.2f}  = N/A (ningun caso supera el umbral)")
    if total_psma > 0:
        print(f"  Cobertura (recall) de PSMA a ese umbral: "
              f"{n_conf_ok}/{total_psma} = {cobertura*100:.2f}%")

    # -------- 7.3 Matriz de confusion "de RUTA" (tu logica de despacho) --------
    # Regla: se envia a la ruta PSMA-especializada SII pred=PSMA & conf>umbral.
    #        en caso contrario (pred=FDG, o PSMA con conf baja) -> MODELO CONJUNTO.
    route_psma = mask_conf_psma
    cm_route = np.zeros((2, 2), dtype=int)   # filas: real[FDG,PSMA]; col:[Conjunto, PSMA_esp]
    for t in (0, 1):
        sel = (y_true == t)
        cm_route[t, 1] = int(np.sum(sel & route_psma))
        cm_route[t, 0] = int(np.sum(sel & ~route_psma))
    print_matrix(cm_route, ["FDG", "PSMA"],
                 ["Modelo_conjunto", "Ruta_PSMA_esp"],
                 f"MATRIZ DE RUTA (PSMA-esp sii conf>{conf_threshold:.2f})")
    prec_route = cm_route[1, 1] / cm_route[:, 1].sum() if cm_route[:, 1].sum() > 0 else float('nan')
    print(f"  (precision de la ruta PSMA = {prec_route*100:.2f}%  == PRECISION PSMA @ umbral)")

    # -------- 7.4 Barrido de umbrales (contexto precision/cobertura) --------
    print("\n--- BARRIDO DE UMBRALES (ruta PSMA) ---")
    print(f"{'umbral':>7} {'n_ruta':>7} {'aciertos':>9} {'errores':>8} "
          f"{'precision':>10} {'cobertura':>10}")
    for thr in (0.50, 0.60, conf_threshold, 0.70, 0.80, 0.90):
        m = (y_pred == 1) & (p_psma > thr)
        n = int(m.sum())
        ok = int(np.sum(m & (y_true == 1)))
        er = n - ok
        pr = f"{ok/n*100:6.2f}%" if n > 0 else "   N/A"
        co = f"{ok/total_psma*100:6.2f}%" if total_psma > 0 else "   N/A"
        star = " *" if abs(thr - conf_threshold) < 1e-9 else ""
        print(f"{thr:>7.2f} {n:>7d} {ok:>9d} {er:>8d} {pr:>10} {co:>10}{star}")

    # -------- 7.5 PNGs --------
    os.makedirs(out_dir, exist_ok=True)
    save_matrix_png(cm, ["FDG", "PSMA"], ["FDG", "PSMA"],
                    "Matriz de confusion (argmax)",
                    os.path.join(out_dir, f"cm_argmax{tag}.png"))
    save_matrix_png(cm_route, ["FDG", "PSMA"], ["Conjunto", "PSMA_esp"],
                    f"Matriz de ruta (conf>{conf_threshold:.2f})",
                    os.path.join(out_dir, f"cm_ruta{tag}.png"), xlabel="Ruta")

    # -------- 7.6 CSV por caso --------
    df_out = pd.DataFrame({
        'filename': df_test['filename'].values if 'filename' in df_test else np.arange(len(y_true)),
        'true_label': y_true,
        'true_tracer': [TRACER_NAMES[t] for t in y_true],
        'pred_label': y_pred,
        'pred_tracer': [TRACER_NAMES[t] for t in y_pred],
        'p_fdg': proba[:, list(rf.classes_).index(0)],
        'p_psma': p_psma,
        'confident_psma': mask_conf_psma,
        'route': np.where(route_psma, 'PSMA_esp', 'Conjunto'),
        'correct': (y_pred == y_true),
    })
    csv_out = os.path.join(out_dir, f"predicciones_por_caso{tag}.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"\n[+] Predicciones por caso guardadas en: {csv_out}")

    # -------- 7.7 Resumen txt --------
    with open(os.path.join(out_dir, f"resumen{tag}.txt"), "w") as f:
        f.write(f"Accuracy global (argmax): {acc*100:.2f}%\n")
        f.write(f"Casos: {len(y_true)} | PSMA reales: {total_psma}\n")
        f.write(f"Matriz confusion (argmax) [FDG,PSMA]x[FDG,PSMA]:\n{cm}\n")
        f.write(f"\nPSMA @ conf>{conf_threshold:.2f}:\n")
        f.write(f"  n_ruta_PSMA={n_conf}  aciertos={n_conf_ok}  errores={n_conf_err}\n")
        f.write(f"  PRECISION_PSMA={prec_conf*100:.2f}%  COBERTURA={cobertura*100:.2f}%\n")

    return dict(accuracy=acc, precision_psma_conf=prec_conf, n_ruta_psma=n_conf,
                aciertos=n_conf_ok, errores=n_conf_err, cobertura=cobertura)


# =====================================================================
# 8. MAIN
# =====================================================================
def main():
    global NNUNET_RAW  # permitimos override por CLI

    ap = argparse.ArgumentParser(
        description="Entrena RF (FDG/PSMA) en un dataset y lo evalua en otro (cross-eval).")
    ap.add_argument('--nnunet-raw', default=NNUNET_RAW)
    ap.add_argument('--train-dataset', default=TRAIN_DATASET)
    ap.add_argument('--test-dataset', default=TEST_DATASET)
    ap.add_argument('--channel-suffix', default=None,
                    help="Fuerza el sufijo del canal PET (p.ej. 0001.nii.gz). "
                         "Si se omite, se autodetecta por dataset desde dataset.json.")
    ap.add_argument('--conf-threshold', type=float, default=CONF_THRESHOLD)
    ap.add_argument('--model-out', default=MODEL_OUT)
    ap.add_argument('--model-in', default=None,
                    help="Si se da y existe, se carga el modelo y se SALTA el entrenamiento.")
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--reuse-features', action='store_true',
                    help="Reusa los CSV de features si ya existen (evita re-extraer).")
    args = ap.parse_args()

    NNUNET_RAW = args.nnunet_raw
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("CLASIFICADOR DE TRAZADOR (FDG/PSMA) — EVALUACION CRUZADA")
    print("=" * 70)
    print(f"NNUNET_RAW    : {NNUNET_RAW}")
    print(f"TRAIN dataset : {args.train_dataset}")
    print(f"TEST  dataset : {args.test_dataset}")
    print(f"Umbral conf.  : {args.conf_threshold}")

    # Sufijo del canal PET (autodetectado por dataset salvo override explicito)
    suf_train = args.channel_suffix or detectar_sufijo_pet(args.train_dataset)
    suf_test = args.channel_suffix or detectar_sufijo_pet(args.test_dataset)
    print(f"Canal PET     : train='{suf_train}'  test='{suf_test}'")

    csv_train = os.path.join(args.out_dir, f"features_{args.train_dataset}.csv")
    csv_test = os.path.join(args.out_dir, f"features_{args.test_dataset}.csv")

    # -------- 8.1 Modelo: cargar o entrenar --------
    if args.model_in and os.path.exists(args.model_in):
        print(f"\n[=] Cargando modelo existente: {args.model_in} (se salta el entrenamiento)")
        bundle = joblib.load(args.model_in)
        rf, feat_cols = bundle['model'], bundle['features']
    else:
        print("\n### PASO 1: ENTRENAMIENTO ###")
        df_train = cargar_o_extraer(args.train_dataset, csv_train, suf_train, args.reuse_features)
        if df_train is None or df_train.empty:
            raise SystemExit("No hay datos de entrenamiento. Revisa rutas/sufijo.")
        rf, feat_cols = entrenar(df_train, args.model_out)

    # -------- 8.2 Evaluacion cruzada --------
    print("\n### PASO 2: EVALUACION EN EL DATASET DE TEST ###")
    df_test = cargar_o_extraer(args.test_dataset, csv_test, suf_test, args.reuse_features)
    if df_test is None or df_test.empty:
        raise SystemExit("No hay datos de test. Revisa rutas/sufijo.")

    res = evaluar(rf, feat_cols, df_test, args.conf_threshold,
                  args.out_dir, tag=f"_{args.test_dataset}")

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Accuracy global (argmax) : {res['accuracy']*100:.2f}%")
    if res['n_ruta_psma'] > 0:
        print(f"  PRECISION PSMA @ {args.conf_threshold:.2f}  : {res['precision_psma_conf']*100:.2f}%  "
              f"({res['aciertos']} aciertos / {res['n_ruta_psma']} enviados; {res['errores']} FDG colados)")
    else:
        print(f"  PRECISION PSMA @ {args.conf_threshold:.2f}  : N/A (ningun caso supera el umbral)")
    print(f"  Resultados en: {args.out_dir}/")


if __name__ == '__main__':
    main()