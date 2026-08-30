"""
nnUNetTr4_Interactive: FASE 2 (aprendizaje interactivo) sobre el linaje 4.

Toma como backbone el L4 de fase 1 (nnUNetTr4_PreTr): UN encoder (2 canales,
CT+PET) + UN decoder cuya cabeza final esta ENSANCHADA a N_LESION + N_ORGAN
canales (2 lesion + 10 organos, dos softmax independientes). La guia interactiva
(prev_mask, dist_fg, dist_bg) NO se apila al stem: se inyecta en cada skip del
decoder via guidance_proj (convs 1x1 zero-init, aditivas), igual que en el
interactivo del linaje 1. Como el L4 es un solo encoder-decoder, la inyeccion es
directa (no hay fusion de dos ramas).

POR QUE el L4 y no el 34c/34cv2:
- En fase 1 el L4 es el mejor (Score 0.668) y el mas ROBUSTO (std entre folds
  0.025 vs 0.066 del 34c) -> menos riesgo sobre un test fijo.
- El 34cv2 (Focal Tversky + blob, sin cabeza PET) REGRESO: su loss precision-
  leaning subio el FN (recall peor), y para una base de fase 2 quieres recall
  (los scribbles quitan FP facil, pero no recuperan lo nunca detectado).
- La organ supervision del L4 (FP bajos) complementa la fase 2: menos FP que
  limpiar, los scribbles se centran en las lesiones dificiles.

WARM START (desde nnUNetTr4_PreTr):
- El decoder se construye YA con N_LESION + N_ORGAN = 12 canales de salida, asi
  que decoder.seg_layers coincide 1:1 con las seg_layers ensanchadas que guarda
  el checkpoint del L4. encoder.* y decoder.* se copian 1:1. guidance_proj.* no
  existe en el L4 -> queda zero-init, de modo que en el paso 0 la red reproduce
  el L4 exactamente (proj suma 0).

Loss = W_LESION * DiceCE(lesion)  +  W_ORGAN * DiceCE_ignore(organos)  +
       W_SCRIBBLE * scribble(lesion)
  - Lesion: Dice+CE ESTANDAR (robusto).  NO Focal Tversky (regreso el recall).
  - Organos: DC_and_CE con ignore_label=99 (se conserva la anatomia -> FP bajos).
  - Scribble: log-loss en los voxeles marcados (fg -> hacia 1, bg -> hacia 0).

DATOS: los mismos del L4 (preprocesado CON organos). Imagen 2 canales (CT=_0000,
PET=_0001). Seg 2 canales: ch0 lesion (0/1), ch1 organos (0..9 / 99).
INFERENCIA: la red devuelve SOLO la lesion del decoder.
"""

import os
import sys
import re
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch._dynamo import OptimizedModule
from torch.optim.lr_scheduler import _LRScheduler
from scipy.ndimage import label as cc_label

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.paths import nnUNet_results

torch._dynamo.config.suppress_errors = True
torch.compiler.disable()

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from simulate_scribbles import simulate_scribble_from_label

ORGAN_IGNORE    = 99
N_ORGAN_CLASSES = 10      # 0 = fondo + 9 organos

# ==========================================================================
#  FOLD (datos + checkpoint del L4). None -> fold de la CLI ; entero -> forzado.
# ==========================================================================
FOLD_OVERRIDE = None


# ============================================================================
#  RED: encoder + decoder (cabeza ensanchada lesion+organos) + guidance_proj
# ============================================================================
class InteractiveL4Net(nn.Module):
    def __init__(self, base_unet, n_image, guidance_channels, n_lesion, n_organ,
                 enable_deep_supervision=True):
        super().__init__()
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.deep_supervision = enable_deep_supervision
        self.decoder.deep_supervision = enable_deep_supervision
        self.n_image = n_image                    # 2: CT, PET
        self.guidance_channels = guidance_channels  # 3: prev, dist_fg, dist_bg
        self.n_lesion = n_lesion                  # 2
        self.n_organ = n_organ                    # 10

        ch = self.encoder.output_channels
        self.guidance_proj = nn.ModuleList([
            nn.Conv3d(guidance_channels, c, kernel_size=1) for c in ch
        ])
        for proj in self.guidance_proj:
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _split(self, t):
        return t[:, :self.n_lesion], t[:, self.n_lesion:]

    def forward(self, x):
        img = x[:, :self.n_image]
        aux = x[:, self.n_image:self.n_image + self.guidance_channels]

        skips = self.encoder(img)
        skips = [
            s + self.guidance_proj[i](
                F.interpolate(aux, size=s.shape[2:], mode='trilinear', align_corners=False))
            for i, s in enumerate(skips)
        ]
        out = self.decoder(skips)                 # lista (deep sup) o tensor, 12 canales

        if self.training:
            if isinstance(out, (list, tuple)):
                les = [self._split(t)[0] for t in out]
                org = [self._split(t)[1] for t in out]
            else:
                les_t, org_t = self._split(out)
                les, org = les_t, org_t
            return les, org
        else:
            # eval: SOLO lesion, preservando estructura (lista si DS on, tensor si off)
            if isinstance(out, (list, tuple)):
                return [self._split(t)[0] for t in out]
            return self._split(out)[0]


# ============================================================================
#  LR SCHEDULER
# ============================================================================
class PolyLRWarmupMultiGroup(_LRScheduler):
    def __init__(self, optimizer, max_steps, warmup_epochs=0, exponent=0.9, current_step=None):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.exponent = exponent
        self.base_lrs_ = [g['lr'] for g in optimizer.param_groups]
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        if self.warmup_epochs > 0 and current_step < self.warmup_epochs:
            factor = (current_step + 1) / self.warmup_epochs
        else:
            t = current_step - self.warmup_epochs
            T = max(self.max_steps - self.warmup_epochs, 1)
            factor = (1 - t / T) ** self.exponent
        for g, base in zip(self.optimizer.param_groups, self.base_lrs_):
            g['lr'] = base * factor


# ============================================================================
#  TRAINER
# ============================================================================
class nnUNetTr4_Interactive2(nnUNetTrainer):

    N_IMAGE = 2       # CT, PET
    N_GUIDANCE = 3    # prev_mask, dist_fg, dist_bg
    N_LESION = 2

    W_LESION = 1.0
    W_ORGAN = 0.5     # auxiliar: mantiene la anatomia (FP bajos) sin dominar en fase 2
    W_SCRIBBLE = 0.5

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # -------- FOLD (datos + checkpoint del L4) --------
        if FOLD_OVERRIDE is not None:
            self.fold = FOLD_OVERRIDE
            if getattr(self, 'output_folder_base', None) is not None:
                self.output_folder = os.path.join(self.output_folder_base, f'fold_{self.fold}')
        self.pretrained_fold = self.fold

        # checkpoint del L4 de fase 1: MISMO dataset que el -d del comando y raiz
        # de nnUNet_results (respeta tu export; sin rutas ni dataset hardcodeados).
        dataset_name = self.plans_manager.dataset_name        # p.ej. 'Dataset003_...'
        self.path_pretrained_l4 = os.path.join(
            nnUNet_results, dataset_name,
            "nnUNetTr4_PreTr__nnUNetPlans__3d_fullres",
            f"fold_{self.pretrained_fold}", "checkpoint_best.pth")

        self.num_epochs = 500

        # --- codificacion de la posicion del scribble: heatmap GAUSSIANO ---
        #   exp(-d^2 / 2 sigma^2), pico 1 (sin normalizar a volumen unidad).
        self.dist_sigma = 5.0

        self.scribble_strategies = ("centerline", "boundary", "random")
        self.max_scribble_pts = 16
        self.use_gpu_clicks = False
        self.click_pool_k = 5

        self.initial_lr = 1e-3
        self.encoder_lr_mult = 0.1
        self.warmup_epochs = 10
        self.grad_clip = 3.0

        # nº de interacciones (0..5): mayor masa en 0 -> conserva la seg. inicial
        self.click_probs = torch.tensor([0.24, 0.18, 0.13, 0.09, 0.11, 0.25])
        self.val_num_steps = len(self.click_probs)
        self.val_auc_include_initial = True

        self._organ_loss = None
        self._dist_margin = self._guidance_margin()
        self._step_tp = self._step_fp = self._step_fn = None
        self._dbg = None

    # ---------------- utilidades ----------------
    def _guidance_margin(self):
        # radio de la ventana donde se calcula el campo (~3 sigma)
        return int(np.ceil(3.0 * self.dist_sigma))

    def _unwrap(self):
        net = self.network.module if self.is_ddp else self.network
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        return net

    def _do_i_compile(self):
        return False

    @staticmethod
    def _lesion_target(target):
        if isinstance(target, (list, tuple)):
            return [t[:, 0:1] for t in target]
        return target[:, 0:1]

    @staticmethod
    def _organ_target(target):
        if isinstance(target, (list, tuple)):
            return [t[:, 1:2] for t in target]
        return target[:, 1:2]

    def _fg_prob(self, preds):
        # preds: salida de lesion (lista o tensor, 2 canales)
        main = preds[0] if isinstance(preds, (list, tuple)) else preds
        return torch.softmax(main.float(), dim=1)[:, 1:2]

    def _new_dbg(self):
        return dict(n_fg=0, n_bg=0, n_bg_forced=0, fp_sum=0, fn_sum=0, n_steps=0)

    # ---------------- losses ----------------
    def _build_loss(self):
        # self.loss = loss de LESION (Dice+CE estandar), envuelta en deep supervision.
        lesion = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {}, weight_ce=1, weight_dice=1, ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss)
        if self.enable_deep_supervision:
            ds = self._get_deep_supervision_scales()
            w = np.array([1 / (2 ** i) for i in range(len(ds))])
            w[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            w = w / w.sum()
            return DeepSupervisionWrapper(lesion, w)
        return lesion

    def _build_organ_loss(self):
        organ = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {}, weight_ce=1, weight_dice=1, ignore_label=ORGAN_IGNORE,
            dice_class=MemoryEfficientSoftDiceLoss)
        if self.enable_deep_supervision:
            ds = self._get_deep_supervision_scales()
            w = np.array([1 / (2 ** i) for i in range(len(ds))])
            w[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            w = w / w.sum()
            return DeepSupervisionWrapper(organ, w)
        return organ

    def _scribble_loss(self, final_lesion, fg_mask, bg_mask):
        fg_prob = self._fg_prob(final_lesion).clamp(1e-6, 1 - 1e-6)
        terms, n = 0.0, 0
        for b in range(fg_mask.shape[0]):
            p = fg_prob[b, 0]; fs = fg_mask[b, 0]; bs = bg_mask[b, 0]
            if fs.any():
                terms = terms - torch.log(p[fs]).mean(); n += 1
            if bs.any():
                terms = terms - torch.log(1 - p[bs]).mean(); n += 1
        return terms / n if n > 0 else torch.zeros((), device=self.device)

    # ---------------- guia / campo de scribbles EN GPU ----------------
    def _add_bump(self, heat, b, coords):
        if coords.numel() == 0:
            return
        D, H, W = heat.shape[2:]
        m = self._dist_margin
        z0 = max(int(coords[:, 0].min()) - m, 0); z1 = min(int(coords[:, 0].max()) + m + 1, D)
        y0 = max(int(coords[:, 1].min()) - m, 0); y1 = min(int(coords[:, 1].max()) + m + 1, H)
        x0 = max(int(coords[:, 2].min()) - m, 0); x1 = min(int(coords[:, 2].max()) + m + 1, W)

        dev = heat.device
        zz = torch.arange(z0, z1, device=dev).view(-1, 1, 1)
        yy = torch.arange(y0, y1, device=dev).view(1, -1, 1)
        xx = torch.arange(x0, x1, device=dev).view(1, 1, -1)

        d2 = torch.full((z1 - z0, y1 - y0, x1 - x0), float('inf'), device=dev)
        cz, cy, cx = coords[:, 0], coords[:, 1], coords[:, 2]
        for k in range(coords.shape[0]):
            d2 = torch.minimum(d2, (zz - cz[k]) ** 2 + (yy - cy[k]) ** 2 + (xx - cx[k]) ** 2)

        # heatmap gaussiano (pico 1, sin normalizar a volumen unidad)
        bump = torch.exp(-d2.float() / (2.0 * self.dist_sigma ** 2))
        win = heat[b, 0, z0:z1, y0:y1, x0:x1]
        heat[b, 0, z0:z1, y0:y1, x0:x1] = torch.maximum(win, bump)

    def _click_gpu(self, err_mask):
        f = err_mask.float()[None, None]
        k = self.click_pool_k
        dens = F.avg_pool3d(f, kernel_size=k, stride=1, padding=k // 2) * f
        idx = torch.argmax(dens.flatten())
        D, H, W = err_mask.shape
        z = idx // (H * W); y = (idx % (H * W)) // W; x = idx % W
        return torch.stack([z, y, x]).view(1, 3).long()

    def _sample_scribble(self, err_mask):
        if self.use_gpu_clicks:
            return self._click_gpu(err_mask)
        err_np = err_mask.detach().cpu().numpy().astype(np.uint8)
        dev = err_mask.device
        best_d, best_area, best_comp = None, 0, None
        for d in range(err_np.shape[0]):
            sl = err_np[d]
            if sl.sum() == 0:
                continue
            lab2d, n = cc_label(sl, structure=np.ones((3, 3), dtype=np.uint8))
            if n == 0:
                continue
            sizes = np.bincount(lab2d.ravel()); sizes[0] = 0
            lid = int(sizes.argmax()); area = int(sizes[lid])
            if area > best_area:
                best_area = area; best_d = d
                best_comp = (lab2d == lid).astype(np.uint8)
        if best_comp is None:
            return torch.empty((0, 3), dtype=torch.long, device=dev)
        strat = self.scribble_strategies[np.random.randint(len(self.scribble_strategies))]
        yx = self._draw_2d_scribble(best_comp, strat)
        if len(yx) == 0:
            ys, xs = np.nonzero(best_comp)
            yx = [(int(ys[0]), int(xs[0]))]
        coords = np.array([[best_d, y, x] for (y, x) in yx], dtype=int)
        coords = coords[err_np[coords[:, 0], coords[:, 1], coords[:, 2]] > 0]
        if len(coords) == 0:
            return torch.empty((0, 3), dtype=torch.long, device=dev)
        if len(coords) > self.max_scribble_pts:
            sel = np.random.choice(len(coords), self.max_scribble_pts, replace=False)
            coords = coords[sel]
        return torch.as_tensor(coords, dtype=torch.long, device=dev)

    def _draw_2d_scribble(self, slice_mask, strategy):
        from skimage.morphology import skeletonize
        from skimage.segmentation import find_boundaries
        from skimage.draw import line as sk_line
        m = slice_mask.astype(np.uint8)
        if strategy == "centerline":
            skel = skeletonize(m).astype(np.uint8)
            pts = np.argwhere(skel)
            if len(pts) < 2:
                pts = np.argwhere(m)
            if len(pts) > 10:
                s = int(len(pts) * 0.1); e = int(len(pts) * 0.9); pts = pts[s:e]
            return [(int(y), int(x)) for y, x in pts if m[y, x]]
        if strategy == "boundary":
            bnd = find_boundaries(m, mode='inner').astype(np.uint8)
            coords = np.argwhere(bnd)
            if len(coords) < 2:
                return [(int(y), int(x)) for y, x in np.argwhere(m)]
            start = tuple(coords[np.random.randint(len(coords))])
            visited = {start}; cur = start; out = [start]
            for _ in range(max(2, int(0.2 * len(coords)))):
                y, x = cur; nxt = None
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx_ = y + dy, x + dx
                        if (0 <= ny < bnd.shape[0] and 0 <= nx_ < bnd.shape[1]
                                and bnd[ny, nx_] and (ny, nx_) not in visited):
                            nxt = (ny, nx_); break
                    if nxt:
                        break
                if nxt is None:
                    break
                visited.add(nxt); out.append(nxt); cur = nxt
            return [(int(y), int(x)) for (y, x) in out]
        coords = np.argwhere(m)
        if len(coords) < 2:
            return [(int(y), int(x)) for y, x in coords]
        p1 = coords[np.random.randint(len(coords))]; p2 = coords[np.random.randint(len(coords))]
        rr, cc = sk_line(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
        valid = (rr >= 0) & (rr < m.shape[0]) & (cc >= 0) & (cc < m.shape[1])
        rr, cc = rr[valid], cc[valid]
        return [(int(r), int(c)) for r, c in zip(rr, cc) if m[r, c]]

    def _update_interaction(self, hard, gt_bin, dist_fg, dist_bg, fg_mask, bg_mask):
        B = hard.shape[0]
        fp = hard & (~gt_bin); fn = (~hard) & gt_bin
        dbg = self._dbg
        for b in range(B):
            fpb, fnb = fp[b, 0], fn[b, 0]
            n_fp = int(fpb.sum().item()); n_fn = int(fnb.sum().item())
            if dbg is not None:
                dbg['fp_sum'] += n_fp; dbg['fn_sum'] += n_fn; dbg['n_steps'] += 1
            is_bg = n_fp > n_fn
            err = fpb if is_bg else fnb
            forced_bg = False
            if err.any():
                coords = self._sample_scribble(err)
            else:
                bgv = torch.nonzero(~gt_bin[b, 0], as_tuple=False)
                if bgv.numel() == 0:
                    continue
                coords = bgv[torch.randint(0, bgv.shape[0], (1,), device=bgv.device)]
                is_bg = True; forced_bg = True
            if coords.numel() == 0:
                continue
            if is_bg:
                self._add_bump(dist_bg, b, coords)
                bg_mask[b, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = True
                if dbg is not None:
                    dbg['n_bg'] += 1; dbg['n_bg_forced'] += int(forced_bg)
            else:
                self._add_bump(dist_fg, b, coords)
                fg_mask[b, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = True
                if dbg is not None:
                    dbg['n_fg'] += 1

    # ---------------- metricas (sobre la lesion) ----------------
    def _hard_tp_fp_fn(self, lesion_output, target_lesion_list):
        out = lesion_output[0] if isinstance(lesion_output, (list, tuple)) else lesion_output
        target = target_lesion_list[0]
        axes = [0] + list(range(2, out.ndim))
        seg = out.argmax(1)[:, None]
        pred_onehot = torch.zeros(out.shape, device=out.device, dtype=torch.float32)
        pred_onehot.scatter_(1, seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_onehot, target, axes=axes, mask=None)
        return tp[1:], fp[1:], fn[1:]

    # ---------------- construccion ----------------
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        # encoder de 2 canales; decoder con salida ENSANCHADA a N_LESION + N_ORGAN
        # (para casar 1:1 con las seg_layers del checkpoint del L4).
        base = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager,
            nnUNetTr4_Interactive2.N_IMAGE,
            nnUNetTr4_Interactive2.N_LESION + N_ORGAN_CLASSES,
            enable_deep_supervision)
        return InteractiveL4Net(
            base, n_image=nnUNetTr4_Interactive2.N_IMAGE,
            guidance_channels=nnUNetTr4_Interactive2.N_GUIDANCE,
            n_lesion=nnUNetTr4_Interactive2.N_LESION, n_organ=N_ORGAN_CLASSES,
            enable_deep_supervision=enable_deep_supervision)

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._unwrap()
        mod.deep_supervision = enabled
        mod.decoder.deep_supervision = enabled

    # ---------------- warm start desde el L4 ----------------
    def _transfer_l4(self, path):
        net = self._unwrap()
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        old = {k.replace('module.', ''): v for k, v in ckpt['network_weights'].items()}
        tgt = net.state_dict()
        new_sd, n_copy, n_skip, n_mm = {}, 0, 0, 0
        for k, v in old.items():
            if not (k.startswith('encoder.') or k.startswith('decoder.')):
                continue
            if k not in tgt:
                n_skip += 1; continue
            if v.shape == tgt[k].shape:
                new_sd[k] = v; n_copy += 1
            else:
                n_mm += 1
                self.print_to_log_file(
                    f"  AVISO shape mismatch (omitido): {k}  L4 {tuple(v.shape)} vs fase2 {tuple(tgt[k].shape)}")
        net.load_state_dict(new_sd, strict=False)
        n_proj = sum(p.numel() for p in net.guidance_proj.parameters())
        self.print_to_log_file(
            f"Warm start desde L4 (fold {self.pretrained_fold}) -> copiados 1:1: {n_copy}  "
            f"omitidos: {n_skip}  mismatch: {n_mm}")
        self.print_to_log_file(
            f"  guidance_proj zero-init ({n_proj} params): en el paso 0 la red = L4.")
        if n_copy == 0:
            raise RuntimeError("Warm start copio 0 tensores: revisa que el checkpoint sea el del L4 "
                               "(seg_layers ensanchadas a 12 canales).")

    def configure_optimizers(self):
        net = self._unwrap()
        rest = [p for n, p in net.named_parameters() if not n.startswith('encoder.')]
        enc = [p for n, p in net.named_parameters() if n.startswith('encoder.')]
        optimizer = torch.optim.SGD(
            [{'params': rest, 'lr': self.initial_lr},
             {'params': enc, 'lr': self.initial_lr * self.encoder_lr_mult}],
            lr=self.initial_lr, momentum=0.99, nesterov=True, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRWarmupMultiGroup(
            optimizer, self.num_epochs, warmup_epochs=self.warmup_epochs, exponent=0.9)
        return optimizer, lr_scheduler

    def initialize(self):
        super().initialize()
        self._dist_margin = self._guidance_margin()
        self.print_to_log_file(
            f"FASE 2 interactiva sobre L4.  Encoder {self.N_IMAGE} ch (CT, PET) + "
            f"guidance_proj (prev, dist_fg, dist_bg).  Cabeza ensanchada "
            f"{self.N_LESION}+{N_ORGAN_CLASSES}.  Scribbles: heatmap gaussiano "
            f"(sigma {self.dist_sigma:g}).")
        _src = "FOLD_OVERRIDE" if FOLD_OVERRIDE is not None else "CLI"
        self.print_to_log_file(f"Fold efectivo (datos + L4): {self.fold}  [fuente: {_src}]")
        if os.path.exists(self.path_pretrained_l4):
            self._transfer_l4(self.path_pretrained_l4)
        else:
            cand = glob.glob(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(self.path_pretrained_l4))),
                '*', 'fold_*', '*.pth'))
            msg = (f"\n{'='*70}\ncheckpoint del L4 NO existe:\n  {self.path_pretrained_l4}\n"
                   f"Abortando. Encontrados:\n" + ("\n".join(f"  {c}" for c in sorted(cand)[:20]) or "  (ninguno)")
                   + f"\n{'='*70}")
            self.print_to_log_file(msg)
            raise FileNotFoundError(msg)
        self.network.to(self.device)

    # ---------------- train ----------------
    def train_step(self, batch):
        if self._organ_loss is None:
            self._organ_loss = self._build_organ_loss()

        data_full = batch['data'].to(self.device, non_blocking=True)
        data = data_full[:, :self.N_IMAGE]                     # CT, PET
        target = batch['target']
        target_list = target if isinstance(target, list) else [target]
        target_list = [t.to(self.device, non_blocking=True) for t in target_list]
        tgt_les = self._lesion_target(target_list)
        tgt_org = self._organ_target(target_list)
        gt_bin = (tgt_les[0] > 0)

        B, _, D, H, W = data.shape
        prev_mask = torch.zeros((B, 1, D, H, W), device=self.device)
        dist_fg = torch.zeros((B, 1, D, H, W), device=self.device)
        dist_bg = torch.zeros((B, 1, D, H, W), device=self.device)
        fg_mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=self.device)
        bg_mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=self.device)

        n_clicks = torch.multinomial(self.click_probs, 1).item()
        num_sim_steps = n_clicks + 1

        self.network.eval()
        with torch.no_grad(), (autocast(self.device.type, enabled=True)
                               if self.device.type == 'cuda' else dummy_context()):
            for _ in range(num_sim_steps - 1):
                net_input = torch.cat([data, prev_mask, dist_fg, dist_bg], dim=1)
                lesion = self.network(net_input)               # eval -> solo lesion
                prev_mask = self._fg_prob(lesion)
                hard = (prev_mask > 0.5)
                self._update_interaction(hard, gt_bin, dist_fg, dist_bg, fg_mask, bg_mask)

        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            net_input = torch.cat([data, prev_mask, dist_fg, dist_bg], dim=1)
            les_out, org_out = self.network(net_input)
            l_les = self.loss(les_out, tgt_les)
            l_org = self._organ_loss(org_out, tgt_org)
            l = self.W_LESION * l_les + self.W_ORGAN * l_org
            if self.W_SCRIBBLE > 0:
                l = l + self.W_SCRIBBLE * self._scribble_loss(les_out, fg_mask, bg_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
        return {'loss': l_les.detach().cpu().numpy()}

    # ---------------- validacion interactiva ----------------
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        n = self.val_num_steps
        self._step_tp = torch.zeros(n, device=self.device)
        self._step_fp = torch.zeros(n, device=self.device)
        self._step_fn = torch.zeros(n, device=self.device)
        self._dbg = self._new_dbg()

    def validation_step(self, batch):
        data_full = batch['data'].to(self.device, non_blocking=True)
        data = data_full[:, :self.N_IMAGE]
        target = batch['target']
        target_list = target if isinstance(target, list) else [target]
        target_list = [t.to(self.device, non_blocking=True) for t in target_list]
        tgt_les = self._lesion_target(target_list)
        gt_bin = (tgt_les[0] > 0)

        B, _, D, H, W = data.shape
        prev_mask = torch.zeros((B, 1, D, H, W), device=self.device)
        dist_fg = torch.zeros((B, 1, D, H, W), device=self.device)
        dist_bg = torch.zeros((B, 1, D, H, W), device=self.device)
        fg_mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=self.device)
        bg_mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=self.device)

        per_tp, per_fp, per_fn = [], [], []
        final_lesion = None
        with torch.no_grad(), (autocast(self.device.type, enabled=True)
                               if self.device.type == 'cuda' else dummy_context()):
            for step in range(self.val_num_steps):
                net_input = torch.cat([data, prev_mask, dist_fg, dist_bg], dim=1)
                lesion = self.network(net_input)
                final_lesion = lesion
                tp, fp, fn = self._hard_tp_fp_fn(lesion, tgt_les)
                self._step_tp[step] += tp.sum(); self._step_fp[step] += fp.sum(); self._step_fn[step] += fn.sum()
                if step > 0 or self.val_auc_include_initial:
                    per_tp.append(tp.detach().cpu().numpy())
                    per_fp.append(fp.detach().cpu().numpy())
                    per_fn.append(fn.detach().cpu().numpy())
                prev_mask = self._fg_prob(lesion)
                hard = (prev_mask > 0.5)
                if step < self.val_num_steps - 1:
                    self._update_interaction(hard, gt_bin, dist_fg, dist_bg, fg_mask, bg_mask)
            l = self.loss(final_lesion, tgt_les)

        return {'loss': l.detach().cpu().numpy(),
                'tp_hard': np.mean(per_tp, axis=0),
                'fp_hard': np.mean(per_fp, axis=0),
                'fn_hard': np.mean(per_fn, axis=0)}

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        tp, fp, fn = self._step_tp, self._step_fp, self._step_fn
        if self.is_ddp:
            import torch.distributed as dist
            dist.all_reduce(tp); dist.all_reduce(fp); dist.all_reduce(fn)
        if not self.is_ddp or self.local_rank == 0:
            d = (2 * tp / (2 * tp + fp + fn + 1e-8)).detach().cpu().numpy()
            auc = float(d.mean())
            curve = "  ".join([f"s{i + 1}:{m:.3f}" for i, m in enumerate(d)])
            self.print_to_log_file(f"AUC-DICE(nnUNet-agg): {auc:.4f} | curva {curve}")
            g = self._dbg
            if g['n_steps'] > 0:
                self.print_to_log_file(
                    f"  [errores] FP/paso={g['fp_sum'] / g['n_steps']:8.1f}  "
                    f"FN/paso={g['fn_sum'] / g['n_steps']:8.1f}  -> {g['n_fg']} FG / {g['n_bg']} BG scribbles")