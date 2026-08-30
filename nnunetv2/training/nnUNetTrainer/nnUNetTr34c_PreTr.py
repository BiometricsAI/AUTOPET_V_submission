"""
nnUNetTr34c_PreTr: FASE 1, linaje 34 MEJORADO ("c" = organ head ACOPLADA).

Objetivo: superar al linaje 4. El L34 original perdia contra el L4 no por el
encoder dual, sino porque su organ supervision estaba DESACOPLADA de la salida de
lesion (decoder_organ separado colgado solo del encoder CT + peso 0.3). Medido:
el L34 tenia MAS falsos positivos que el L3 (que ni siquiera usa organos), senal
de que su supervision de organos estaba practicamente inerte.

QUE CAMBIA respecto al L34 original
-----------------------------------
1) ORGANOS ACOPLADOS A LA FUSION (el arreglo clave). Se ELIMINA el decoder_organ
   separado. En su lugar se ENSANCHA la cabeza final del decoder_joint de
   N_LESION a N_LESION + N_ORGAN canales:
       logits_joint[:, 0:N_LESION]  -> softmax de lesion   (salida PRINCIPAL)
       logits_joint[:, N_LESION: ]  -> softmax de organos  (tarea auxiliar)
   Dos softmax INDEPENDIENTES (un voxel puede ser lesion Y higado a la vez).
   Asi los organos salen del MISMO vector de features fusionado que produce la
   lesion -> acoplamiento fuerte (el mecanismo que hace ganar al L4), pero aqui
   sobre la fusion CT+PET, que ademas trae un encoder CT especializado en
   anatomia. El gradiente de organos regulariza la fusion Y, via la fusion,
   ambos encoders.

2) PESO DE ORGANOS 0.3 -> 1.0 (equitativo con la lesion), como el ganador de
   autoPET IV (organ supervision multi-cabeza con peso equitativo entre heads).

3) Se conserva el encoder DUAL independiente (CT anatomia / PET metabolismo) y la
   cabeza de deteccion PET auxiliar (recall). Es lo que puede darle la ventaja
   sobre el L4: FP bajos (por el acoplamiento) + buena sensibilidad (encoder PET
   especializado + head de deteccion).

Perdida = W_JOINT*Les(joint) + W_ORGAN*Org(joint_head) + W_PET*Les(pet)
  Les = base Dice+CE (deep supervision de nnU-Net).
  Org = DC_and_CE(ignore_label=99) envuelto en deep supervision.
  W_JOINT = W_ORGAN = 1.0 (equitativo);  W_PET = 0.3 (deteccion auxiliar).

Ventaja de MEMORIA sobre el L34 original: se quita un decoder entero
(decoder_organ) y solo se ensancha una conv final -> ocupa MENOS que el L34.

DATOS: identicos al L34 (preprocesado CON organos, v5). Imagen 2 canales
(CT=_0000, PET=_0001). Seg 2 canales: ch0 lesion (0/1), ch1 organos (0..9 / 99).
INFERENCIA: en eval la red devuelve SOLO los canales de lesion del joint (tensor)
-> sliding-window / nnUNetv2_predict sin cambios.
"""

import os

import numpy as np
import torch
from torch import nn, autocast
from torch.utils.tensorboard import SummaryWriter
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

ORGAN_IGNORE    = 99      # debe coincidir con el del preprocesado
N_ORGAN_CLASSES = 10      # 0 = fondo + 9 organos


# =====================================================================
# --- 1. RED: 2 encoders INDEPENDIENTES + fusion + decoder_joint con
#            CABEZA ENSANCHADA (lesion + organos) + decoder_pet (deteccion)
# =====================================================================
class DualEncoderCoupledOrganNet(nn.Module):
    def __init__(self, base_ct, base_pet, base_joint,
                 n_lesion, n_organ, enable_deep_supervision=True):
        super().__init__()
        # DOS encoders con PESOS INDEPENDIENTES (no compartido)
        self.ct_encoder = base_ct.encoder      # anatomia
        self.pet_encoder = base_pet.encoder    # metabolismo

        # Decoder principal: fusion -> lesion + organos (cabeza ensanchada).
        # base_joint se construyo con num_output = n_lesion + n_organ, asi que su
        # conv final ya saca (n_lesion + n_organ) canales.
        self.decoder_joint = base_joint.decoder
        # Decoder auxiliar: deteccion de lesion desde PET (recall)
        self.decoder_pet = base_pet.decoder

        self.n_lesion = n_lesion
        self.n_organ = n_organ
        self.do_ds = enable_deep_supervision

        # Fusion: concat(CT, PET) + conv 1x1 SIN bias por stage
        self.feature_reducers = nn.ModuleList()
        if hasattr(self.ct_encoder, 'output_channels'):
            channels_per_stage = self.ct_encoder.output_channels
        else:
            raise AttributeError("El encoder de nnU-Net no expone 'output_channels'.")
        for out_channels in channels_per_stage:
            self.feature_reducers.append(
                nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False)
            )

    def _split(self, t):
        # t: [B, n_lesion + n_organ, ...] -> (lesion, organos)
        return t[:, :self.n_lesion], t[:, self.n_lesion:]

    def forward(self, x):
        x_ct = x[:, 0:1]        # CT  (_0000)
        x_pet = x[:, 1:2]       # PET (_0001)

        # encoders SEPARADOS (cada uno sus pesos)
        skips_ct = self.ct_encoder(x_ct)
        skips_pet = self.pet_encoder(x_pet)

        # fusion por proyeccion de canales
        skips_joint = [
            self.feature_reducers[i](torch.cat([s_ct, s_pet], dim=1))
            for i, (s_ct, s_pet) in enumerate(zip(skips_ct, skips_pet))
        ]

        joint = self.decoder_joint(skips_joint)   # lista (deep sup) o tensor

        if self.training:
            pet = self.decoder_pet(skips_pet)     # deteccion (auxiliar), n_lesion canales
            if isinstance(joint, (list, tuple)):
                les = [self._split(t)[0] for t in joint]   # lesion por escala
                org = [self._split(t)[1] for t in joint]   # organos por escala
            else:
                les_t, org_t = self._split(joint)
                les, org = les_t, org_t
            return les, org, pet                  # lesion(joint), organos(joint), lesion(pet)
        else:
            # eval: devolver SOLO los canales de lesion del joint, PRESERVANDO la
            # estructura que espera nnU-Net:
            #   - validacion por epoca (deep supervision ON): LISTA de lesion por
            #     escala. self.loss es el DeepSupervisionWrapper y exige lista,
            #     igual que el target (si no, salta el assert del wrapper).
            #   - inferencia final / sliding-window (deep supervision OFF): TENSOR.
            if isinstance(joint, (list, tuple)):
                return [self._split(t)[0] for t in joint]
            return self._split(joint)[0]


# =====================================================================
# --- 2. ENTRENADOR ---
# =====================================================================
class nnUNetTr34c_PreTr(nnUNetTrainer):

    W_JOINT = 1.0      # lesion desde la fusion (PRINCIPAL)
    W_ORGAN = 1.0      # organos desde la MISMA cabeza de fusion (acoplado, equitativo)
    W_PET   = 0.3      # deteccion de lesion desde PET (auxiliar, recall)

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._organ_loss = None
        if not self.is_ddp or self.local_rank == 0:
            tb_dir = os.path.join(self.output_folder, "tensorboard_logs")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            self.epoch_losses = {'Total': [], 'Joint': [], 'PET': [], 'Organ': []}

    def _do_i_compile(self):
        return False       # red con forward custom y salida tuple: sin compile

    def _build_organ_loss(self):
        organ_dc_ce = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {}, weight_ce=1, weight_dice=1,
            ignore_label=ORGAN_IGNORE,
            dice_class=MemoryEfficientSoftDiceLoss)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(organ_dc_ce, weights)
        return organ_dc_ce

    @staticmethod
    def _organ_target(target):
        if isinstance(target, (list, tuple)):
            return [t[:, 1:2] for t in target]
        return target[:, 1:2]

    @staticmethod
    def _lesion_target(target):
        if isinstance(target, (list, tuple)):
            return [t[:, 0:1] for t in target]
        return target[:, 0:1]

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # cada base aporta un encoder o un decoder; se construyen por separado para
        # que los DOS encoders tengan pesos INDEPENDIENTES (init distinta).
        def base(in_ch, out_ch):
            return nnUNetTrainer.build_network_architecture(
                plans_manager, configuration_manager, in_ch, out_ch,
                enable_deep_supervision)

        base_ct    = base(1, num_output_channels)                       # -> ct_encoder (anatomia)
        base_pet   = base(1, num_output_channels)                       # -> pet_encoder + decoder_pet (lesion)
        # decoder_joint con CABEZA ENSANCHADA: lesion + organos
        base_joint = base(1, num_output_channels + N_ORGAN_CLASSES)     # -> decoder_joint (fusion, lesion+organos)

        return DualEncoderCoupledOrganNet(
            base_ct, base_pet, base_joint,
            n_lesion=num_output_channels, n_organ=N_ORGAN_CLASSES,
            enable_deep_supervision=enable_deep_supervision)

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.decoder_joint.deep_supervision = enabled
        mod.decoder_pet.deep_supervision = enabled
        mod.do_ds = enabled

    def train_step(self, batch: dict) -> dict:
        if self._organ_loss is None:
            self._organ_loss = self._build_organ_loss()

        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        tgt_les = self._lesion_target(target)      # ch0 -> lesion (joint y pet)
        tgt_org = self._organ_target(target)       # ch1 -> organos (cabeza joint)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            les_joint, org_joint, pet_les = self.network(data)
            l_joint = self.loss(les_joint, tgt_les)          # lesion desde la fusion (principal)
            l_organ = self._organ_loss(org_joint, tgt_org)   # organos desde la MISMA cabeza (acoplado)
            l_pet = self.loss(pet_les, tgt_les)              # deteccion desde PET (auxiliar)

            l_total = (self.W_JOINT * l_joint) + (self.W_ORGAN * l_organ) + \
                      (self.W_PET * l_pet)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l_total).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l_total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        if not self.is_ddp or self.local_rank == 0:
            self.epoch_losses['Total'].append(l_total.item())
            self.epoch_losses['Joint'].append(l_joint.item())
            self.epoch_losses['PET'].append(l_pet.item())
            self.epoch_losses['Organ'].append(l_organ.item())

        return {'loss': l_joint.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        # eval -> red devuelve lesion(joint) (tensor); recortar target a lesion (ch0)
        batch = dict(batch)
        batch['target'] = self._lesion_target(batch['target'])
        return super().validation_step(batch)

    def on_train_epoch_end(self, train_outputs: list):
        super().on_train_epoch_end(train_outputs)
        if not self.is_ddp or self.local_rank == 0:
            if len(self.epoch_losses['Total']) > 0:
                m = {k: sum(v) / len(v) for k, v in self.epoch_losses.items() if len(v) > 0}
                self.tb_writer.add_scalar('Losses_Epoch/1_Total', m['Total'], self.current_epoch)
                self.tb_writer.add_scalar('Losses_Epoch/2_Joint_Main', m['Joint'], self.current_epoch)
                self.tb_writer.add_scalar('Losses_Epoch/3_PET_Detect', m['PET'], self.current_epoch)
                self.tb_writer.add_scalar('Losses_Epoch/4_Organ_Aux', m['Organ'], self.current_epoch)
            for key in self.epoch_losses.keys():
                self.epoch_losses[key].clear()