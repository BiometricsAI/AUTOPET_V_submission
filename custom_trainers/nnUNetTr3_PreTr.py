"""
Trainer multitarea LESIONES + ORGANOS para la FASE 1 de autoPET V.
Autocontenido: hereda directamente de nnUNetTrainer.

Dos clases:

  nnUNetTrainerOrgan          L = L_CE + L_Dice + 1.0 * L_organos
                              (supervision de organos SOLA -> es el experimento
                               que toca comparar contra nnUNetTr1_PreTr)

  nnUNetTrainerOrganInstance  L = L_CE + L_Dice + 1.0 * L_organos
                                    + 0.2 * L_instance
                              (anade la loss de instancia; siguiente paso de la
                               ablacion, una vez medida la de arriba)

Con WEIGHT_INSTANCE = 0.0 la loss de instancia NO se calcula: no se paga el coste
de cc3d ni la transferencia GPU->CPU.

REQUISITO PREVIO
----------------
prepare_organ_channel.py, que anade las etiquetas de organos como SEGUNDO canal
de segmentacion en los datos preprocesados:

    seg canal 0 = lesion  (0/1)
    seg canal 1 = organos (0..9, o 99 = caso sin supervision de organos)

El trainer comprueba esto al arrancar y aborta con un mensaje claro si falta.

DISENO: CABEZA COMPARTIDA, NO DECODER SEPARADO
----------------------------------------------
Se ensanchan las convoluciones finales del decoder (decoder.seg_layers) de
N_LESION a N_LESION + N_ORGAN canales:

    logits[:, 0:2 ] -> softmax de lesion
    logits[:, 2:12] -> softmax de organos (fondo + 9 grupos)

Dos softmax INDEPENDIENTES: un voxel puede ser "lesion" Y "higado" a la vez.
Importa, porque TotalSegmentator devuelve el organo completo, tumor incluido.

Por que compartida y no un decoder aparte:
  - Es lo que hizo el trainer ganador de autoPET III: una cabeza auxiliar de
    segmentacion de organos, no una rama de decodificacion independiente.
  - El acoplamiento ES el mecanismo. Con la conv final compartida, los logits de
    lesion salen del MISMO vector de features que debe predecir la identidad
    anatomica. Con un decoder separado el gradiente de organos solo alcanza el
    camino de lesion via encoder: acoplamiento mas debil.
  - Coste despreciable: no arriesga el patch size de 192^3.

CASOS SIN SUPERVISION DE ORGANOS
--------------------------------
Un caso cuyo _organs.nii.gz falte o este corrupto NO se descarta: su etiqueta de
lesion sigue siendo valida. Lleva ORGAN_IGNORE (99) en todo el canal 1 y
MaskedOrganLoss suprime la contribucion de organos solo para el.

Se usa 99 y no -1 a proposito: nnU-Net aplica RemoveLabelTransform(-1, 0) en el
aumentado, asi que un -1 se convertiria en 0 = "fondo de organo", supervision
ERRONEA y peor que ninguna. 99 sobrevive al aumentado (nearest) y cabe en int8.

PESO DE LA TAREA DE ORGANOS = 1.0
---------------------------------
El trainer de referencia aplica ponderacion IGUAL a ambas cabezas. Si se observa
que los organos dominan el gradiente, bajar WEIGHT_ORGAN; el punto de partida
documentado es 1.0.

VALIDACION
----------
La red emite 12 canales pero label_manager.num_segmentation_heads sigue siendo 2.
En lugar de reimplementar validation_step (que pisaria cualquier modificacion de
la clase base), se DELEGA en super() y se recorta:
  - la salida, con un forward hook permanente gobernado por self._lesion_only
  - el target, con un slice del batch antes de llamar a super()
El resultado: pseudo-dice y validacion final se calculan solo sobre lesion, y
todo lo que hayas cambiado en nnUNetTrainer.validation_step sigue ejecutandose.
"""

import glob
import multiprocessing
import os.path as osp

import cc3d
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import join
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Proteccion HPC: 'spawn' en lugar de 'fork' ---
torch.multiprocessing.set_sharing_strategy('file_system')
try:
    multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

ORGAN_IGNORE = 99


# =============================================================================
# 1. LOSS DE INSTANCIA
# =============================================================================
class InstanceAveragedSoftDiceLoss(nn.Module):
    """
    Soft Dice por instancia (estilo blob loss) con el falso positivo global
    repartido proporcionalmente al volumen, y ponderacion mixta.

        TP_i = sum_{v in G_i} p_v
        FP   = sum_{v not in G} p_v
        FP_i = FP * |G_i| / |G|

        Dice_i = (2*TP_i + eps) / (TP_i + |G_i| + FP_i + eps)
        W_i    = w_u * (1/N) + (1 - w_u) * (sqrt(|G_i|) / sum_k sqrt(|G_k|))
        L      = sum_i W_i * (1 - Dice_i)

    Reparto proporcional y no FP entero ni FP/N: FP es un recuento absoluto de
    voxels; meterlo crudo junto a |G_i| mezcla escalas sin relacion y satura el
    termino para lesiones pequenas. Con FP_i proporcional, en el optimo
    Dice_i = 2/(2 + FP/|G|), que depende solo de la CARGA RELATIVA de falsos
    positivos y es identica para una lesion de 50 voxels y una de 50.000.
    Ademas sum_i FP_i = FP, o sea el FP se contabiliza integro.

    Los parches sin lesion NO reciben termino propio: los cubren la CE y el
    batch_dice de nnU-Net. Una penalizacion tipo FP/(FP+kappa) tiene derivada
    kappa/(FP+kappa)^2, que CRECE conforme FP baja, y solo empuja hacia abajo:
    realimenta hacia "predecir cero en todas partes" y colapsa el modelo.
    """

    def __init__(self, smooth: float = 1e-5, w_uniform: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.w_uniform = w_uniform

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = softmax_helper_dim1(x)
        batch_size, num_classes = x.shape[0], x.shape[1]
        losses = []

        for b in range(batch_size):
            for c in range(1, num_classes):
                p = x[b, c]

                # Fast path: sync de un escalar en vez de transferir el parche.
                gt_mask = (y[b, 0] == c)
                if not bool(gt_mask.any()):
                    continue

                with torch.no_grad():
                    t_np = gt_mask.detach().cpu().numpy()
                    labeled, num_features = cc3d.connected_components(
                        t_np, connectivity=26, return_N=True)
                    labeled = labeled.astype(np.int32)   # CUDA no procesa uint16
                    labels_t = torch.from_numpy(labeled).to(p.device, non_blocking=True)
                del t_np, labeled, gt_mask

                if num_features == 0:
                    continue

                labels_flat = labels_t.reshape(-1)
                p_flat = p.reshape(-1)
                valid = labels_flat > 0
                labels_valid = labels_flat[valid].to(torch.int64)
                p_valid = p_flat[valid]
                del labels_t, labels_flat, p_flat, valid

                tp = torch.zeros(num_features + 1, dtype=p_valid.dtype, device=p.device)
                tp = tp.index_add(0, labels_valid, p_valid)[1:]

                with torch.no_grad():
                    sum_gt = torch.bincount(
                        labels_valid, minlength=num_features + 1)[1:].to(p.dtype)
                    share = sum_gt / sum_gt.sum().clamp(min=1.0)
                del labels_valid, p_valid

                fp = torch.clamp(p.sum() - tp.sum(), min=0.0)
                dice_inst = (2.0 * tp + self.smooth) / \
                            (tp + sum_gt + fp * share + self.smooth)

                with torch.no_grad():
                    w_size = torch.sqrt(sum_gt)
                    w_size = w_size / w_size.sum().clamp(min=1e-8)
                    w_unif = torch.full_like(w_size, 1.0 / num_features)
                    w = self.w_uniform * w_unif + (1.0 - self.w_uniform) * w_size

                losses.append((w * (1.0 - dice_inst)).sum())

        if len(losses) == 0:
            return x.sum() * 0.0     # mantiene el grafo para DDP
        return torch.stack(losses).mean()


# =============================================================================
# 2. CONSISTENCIA CON SCRIBBLES (INACTIVA EN FASE 1)
# =============================================================================
class ScribbleConsistencyLoss(nn.Module):
    """
    Obliga a la prediccion a coincidir con la etiqueta del usuario en los voxels
    garabateados. Sin esto los modelos aprenden a ignorar parcialmente la
    interaccion y la pendiente de la curva AUC se aplana.

        L = 0.5 * CE sobre S+  +  0.5 * CE sobre S-

    Sin dilatacion: dilatar un scribble positivo cerca del borde de la lesion
    introduce fondo etiquetado como lesion.
    """

    def __init__(self, fg_class: int = 1):
        super().__init__()
        self.fg_class = fg_class

    def forward(self, logits, scribble_fg, scribble_bg):
        logp = torch.log_softmax(logits, dim=1)
        fg = scribble_fg.to(logp.dtype)
        bg = scribble_bg.to(logp.dtype)
        n_fg, n_bg = fg.sum(), bg.sum()

        l_fg = -(logp[:, self.fg_class:self.fg_class + 1] * fg).sum() / n_fg.clamp(min=1.0)
        l_bg = -(logp[:, 0:1] * bg).sum() / n_bg.clamp(min=1.0)

        has_fg = (n_fg > 0).to(logp.dtype)
        has_bg = (n_bg > 0).to(logp.dtype)
        return (l_fg * has_fg + l_bg * has_bg) / (has_fg + has_bg).clamp(min=1.0)


# =============================================================================
# 3. LOSS DE ORGANOS CON IGNORE POR CASO, SIN SINCRONIZACIONES
# =============================================================================
class MaskedOrganLoss(nn.Module):
    """
    CE + soft Dice sobre la cabeza de organos, respetando ORGAN_IGNORE.

    Por que no DC_and_CE_loss con ignore_label: su forward hace
    `if ... num_fg > 0`, un booleano de Python sobre un tensor de GPU, que fuerza
    un sync GPU->CPU. Con deep supervision son ~6 syncs por iteracion, y cada uno
    vacia la cola asincrona de CUDA. Aqui:
      - CE con reduction='none', enmascarada y normalizada por el recuento de
        voxels validos (clamp a 1). Sin sync, sin NaN.
      - Dice reutilizando MemoryEfficientSoftDiceLoss con su parametro loss_mask,
        que ya esta pensado para esto y no materializa el one-hot en float.

    Parche entero ignorado -> CE 0 y Dice constante, o sea gradiente nulo.
    """

    def __init__(self, ignore_label: int = ORGAN_IGNORE, batch_dice: bool = True,
                 smooth: float = 1e-5, ddp: bool = False):
        super().__init__()
        self.ignore_label = ignore_label
        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1, batch_dice=batch_dice,
            do_bg=False, smooth=smooth, ddp=ddp)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        valid = target != self.ignore_label
        target_clean = torch.where(valid, target, torch.zeros_like(target))

        ce_map = F.cross_entropy(logits, target_clean[:, 0], reduction='none')
        v = valid[:, 0].to(ce_map.dtype)
        ce = (ce_map * v).sum() / v.sum().clamp(min=1.0)

        return ce + self.dc(logits, target_clean, loss_mask=valid)


# =============================================================================
# 4. LOSS MULTITAREA
# =============================================================================
class MultiTaskDeepSupervisionWrapper(nn.Module):
    """
        L = sum_s w_s * DCCE_lesion(s)
          + weight_organ    * sum_s w_s * OrganLoss(s)
          + weight_instance * Instance(escala 0)          [si weight_instance > 0]
          + weight_scribble * Scribble(escala 0)          [fase 2]

    El target de entrenamiento tiene 2 canales: [:, 0:1] lesion, [:, 1:2] organos.

    En validacion la salida llega recortada a los canales de lesion y el target a
    1 canal; el termino de organos se omite automaticamente. La loss de instancia
    tampoco se evalua en validacion, para no pagar cc3d 50 veces por epoca.
    """

    def __init__(self, lesion_loss, organ_loss, instance_loss, ds_weights,
                 n_lesion: int = 2,
                 weight_instance: float = 0.0,
                 weight_organ: float = 1.0,
                 scribble_loss=None, weight_scribble: float = 0.0):
        super().__init__()
        self.lesion_loss = lesion_loss
        self.organ_loss = organ_loss
        self.instance_loss = instance_loss
        self.ds_weights = ds_weights
        self.n_lesion = n_lesion
        self.weight_instance = weight_instance
        self.weight_organ = weight_organ
        self.scribble_loss = scribble_loss
        self.weight_scribble = weight_scribble
        self.last_terms = {}

    def forward(self, output, target, scribble_fg=None, scribble_bg=None):
        if not isinstance(output, (list, tuple)):
            output, target = [output], [target]
            ds_w = [1.0]
        else:
            ds_w = self.ds_weights

        n = self.n_lesion
        # Modo "solo lesion": validacion. La salida ya viene recortada por el hook
        # y el target por el slice del batch.
        lesion_only = (output[0].shape[1] <= n) or (target[0].shape[1] < 2)

        l_les, l_org = 0.0, 0.0
        for i, (o, t) in enumerate(zip(output, target)):
            if ds_w[i] == 0.0:
                continue
            l_les = l_les + ds_w[i] * self.lesion_loss(o[:, :n], t[:, 0:1])
            if not lesion_only:
                l_org = l_org + ds_w[i] * self.organ_loss(o[:, n:], t[:, 1:2])

        l_total = l_les
        self.last_terms = {'lesion': l_les.detach() if torch.is_tensor(l_les) else l_les}

        if not lesion_only:
            l_total = l_total + self.weight_organ * l_org
            self.last_terms['organ'] = l_org.detach() if torch.is_tensor(l_org) else l_org

            if self.weight_instance > 0.0:
                l_inst = self.instance_loss(output[0][:, :n], target[0][:, 0:1])
                l_total = l_total + self.weight_instance * l_inst
                self.last_terms['instance'] = l_inst.detach()

            if self.scribble_loss is not None and self.weight_scribble > 0.0 \
                    and scribble_fg is not None:
                l_scr = self.scribble_loss(output[0][:, :n], scribble_fg, scribble_bg)
                l_total = l_total + self.weight_scribble * l_scr
                self.last_terms['scribble'] = l_scr.detach()

        return l_total


# =============================================================================
# 5. TRAINER: SUPERVISION DE ORGANOS SOLA
# =============================================================================
class nnUNetTr3_PreTr(nnUNetTrainer):
    """
    L = L_CE + L_Dice + 1.0 * (L_CE_organos + L_Dice_organos)

    Es el experimento a comparar contra nnUNetTr1_PreTr: UN solo cambio respecto
    al baseline. Para anadir la loss de instancia, usa nnUNetTrainerOrganInstance.
    """

    N_LESION = 2       # fondo + lesion
    N_ORGAN = 10       # fondo + 9 grupos (bazo, rinones, higado, vejiga, pulmon,
                       #                   cerebro, corazon, estomago, prostata)

    WEIGHT_ORGAN = 1.0     # ponderacion igual entre cabezas
    WEIGHT_INSTANCE = 0.0  # 0.0 -> la loss de instancia ni se calcula
    WEIGHT_SCRIBBLE = 0.0  # fase 2: 0.5
    W_UNIFORM = 0.5

    # ---------------------------------------------------------------- init ---
    def initialize(self):
        if self.was_initialized:
            return
        self._lesion_only = False
        super().initialize()          # red (2 canales), optimizador, loss, DDP
        self._install_organ_head()    # ensancha a 12 y REHACE el optimizador

    def _unwrap(self):
        net = self.network
        if isinstance(net, DDP):
            net = net.module
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        return net

    def _install_organ_head(self):
        """
        Ensancha decoder.seg_layers de N_LESION a N_LESION + N_ORGAN canales.

        Se hace DESPUES de super().initialize(), por eso hay que rehacer el
        optimizador: los pesos nuevos no estarian en sus grupos de parametros.
        Con DDP hay que re-envolver, porque el reducer fija su lista de parametros
        al construirse y no veria los nuevos.
        """
        net = self._unwrap()
        n_tot = self.N_LESION + self.N_ORGAN

        new_layers = nn.ModuleList()
        for old in net.decoder.seg_layers:
            conv = type(old)(
                old.in_channels, n_tot,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, dilation=old.dilation,
                bias=old.bias is not None)
            # Los canales de lesion heredan la init original; los de organo se
            # quedan con la init por defecto del conv nuevo.
            with torch.no_grad():
                conv.weight[:self.N_LESION].copy_(old.weight)
                if old.bias is not None:
                    conv.bias[:self.N_LESION].copy_(old.bias)
            new_layers.append(conv)

        net.decoder.seg_layers = new_layers.to(self.device)

        # Mismo orden que nnUNetTrainer.initialize: compile -> optimizador -> DDP
        if self._do_i_compile():
            net = torch.compile(net)
        self.network = net
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        if self.is_ddp:
            self.network = DDP(self.network, device_ids=[self.local_rank])

        # Hook permanente: recorta la salida a la cabeza de lesion cuando
        # self._lesion_only esta activo (validacion). Se registra UNA vez para no
        # provocar recompilaciones de torch.compile.
        n = self.N_LESION

        def _slice_to_lesion(module, inputs, output):
            if not self._lesion_only:
                return output
            if isinstance(output, (list, tuple)):
                return [o[:, :n] for o in output]
            return output[:, :n]

        self.network.register_forward_hook(_slice_to_lesion)

        self.print_to_log_file(
            f'Cabeza multitarea: {self.N_LESION} canales de lesion + '
            f'{self.N_ORGAN} de organos = {n_tot}.  '
            f'WEIGHT_ORGAN={self.WEIGHT_ORGAN}  WEIGHT_INSTANCE={self.WEIGHT_INSTANCE}')

    # ------------------------------------------------- comprobacion de datos --
    def on_train_start(self):
        super().on_train_start()
        self._check_organ_channel()

    def _check_organ_channel(self):
        """
        Aborta pronto y con mensaje claro si falta el canal de organos.
        Soporta blosc2 (nnU-Net 2.6+), .npy desempaquetado y .npz.
        """
        folder = self.preprocessed_dataset_folder
        n_ch, ref = None, None

        b2nd = sorted(glob.glob(join(folder, '*_seg.b2nd')))
        if b2nd:
            try:
                import blosc2
                n_ch = blosc2.open(b2nd[0], mode='r').shape[0]
                ref = osp.basename(b2nd[0])
            except Exception as e:
                self.print_to_log_file(f'AVISO: no se pudo leer {b2nd[0]}: {e!r}')
                return

        if n_ch is None:
            npy = sorted(glob.glob(join(folder, '*_seg.npy')))
            if npy:
                n_ch = np.load(npy[0], mmap_mode='r').shape[0]
                ref = osp.basename(npy[0])

        if n_ch is None:
            npz = sorted(glob.glob(join(folder, '*.npz')))
            if npz:
                with np.load(npz[0]) as f:
                    n_ch = f['seg'].shape[0]
                ref = osp.basename(npz[0])

        if n_ch is None:
            self.print_to_log_file('AVISO: no se pudo comprobar el canal de organos.')
            return

        if n_ch < 2:
            raise RuntimeError(
                f'\n{"="*70}\n'
                f'{ref} tiene {n_ch} canal(es) de segmentacion; se esperan 2 '
                f'(lesion + organos).\n'
                f'Ejecuta prepare_organ_channel.py sobre {folder} antes de '
                f'entrenar con este trainer.\n{"="*70}')

        self.print_to_log_file(f'Canal de organos presente ({n_ch} canales de seg).')

    # ---------------------------------------------------------------- loss ---
    def _build_loss(self):
        dice_kwargs = {'batch_dice': self.configuration_manager.batch_dice,
                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}

        lesion_loss = DC_and_CE_loss(
            dice_kwargs, {}, weight_ce=1, weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss)

        organ_loss = MaskedOrganLoss(
            ignore_label=ORGAN_IGNORE,
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5, ddp=self.is_ddp)

        if self._do_i_compile():
            lesion_loss.dc = torch.compile(lesion_loss.dc)
            organ_loss.dc = torch.compile(organ_loss.dc)

        instance_loss = InstanceAveragedSoftDiceLoss(
            smooth=1e-5, w_uniform=self.W_UNIFORM) \
            if self.WEIGHT_INSTANCE > 0.0 else None

        scribble_loss = ScribbleConsistencyLoss(fg_class=1) \
            if self.WEIGHT_SCRIBBLE > 0.0 else None

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
        else:
            weights = np.array([1.0])

        return MultiTaskDeepSupervisionWrapper(
            lesion_loss=lesion_loss,
            organ_loss=organ_loss,
            instance_loss=instance_loss,
            ds_weights=weights,
            n_lesion=self.N_LESION,
            weight_instance=self.WEIGHT_INSTANCE,
            weight_organ=self.WEIGHT_ORGAN,
            scribble_loss=scribble_loss,
            weight_scribble=self.WEIGHT_SCRIBBLE)

    # ---------------------------------------------------------- validacion ---
    def validation_step(self, batch: dict) -> dict:
        """
        Delega en la clase base recortando entrada y salida a la cabeza de lesion.
        No se reimplementa nada: cualquier modificacion tuya en
        nnUNetTrainer.validation_step (MetaLogger, etc.) sigue ejecutandose.
        """
        tgt = batch['target']
        batch = dict(batch)
        batch['target'] = [t[:, 0:1] for t in tgt] if isinstance(tgt, (list, tuple)) \
            else tgt[:, 0:1]

        self._lesion_only = True
        try:
            return super().validation_step(batch)
        finally:
            self._lesion_only = False

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        El predictor de nnU-Net hace argmax sobre TODOS los canales de salida.
        Con 12 canales las clases de organo ganarian voxels.
        """
        self._lesion_only = True
        try:
            super().perform_actual_validation(save_probabilities)
        finally:
            self._lesion_only = False

    # ------------------------------------------------------------- logging ---
    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        terms = getattr(self.loss, 'last_terms', {})
        if terms and (not self.is_ddp or self.local_rank == 0):
            txt = '  '.join(f'{k}={float(v):.4f}' for k, v in terms.items())
            self.print_to_log_file(f'terminos: {txt}')