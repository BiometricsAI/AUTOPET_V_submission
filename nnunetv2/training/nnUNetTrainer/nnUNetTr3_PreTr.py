import os
import torch
from torch import nn, autocast
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.helpers import dummy_context
from torch._dynamo import OptimizedModule

# =====================================================================
# --- 1. ARQUITECTURA DE LA RED MULTIMODAL (Sin Contrastive) ---
# =====================================================================
class PETCTSharedEncoderNet(nn.Module):
    def __init__(self, base_unet, num_classes, enable_deep_supervision=True):
        super().__init__()
        self.encoder = base_unet.encoder
        
        # Tres caminos de decodificación paralelos
        self.decoder_pet = deepcopy(base_unet.decoder)
        self.decoder_ct = deepcopy(base_unet.decoder)
        self.decoder_joint = deepcopy(base_unet.decoder)
        
        self.do_ds = enable_deep_supervision
        self.feature_reducers = nn.ModuleList()
        
        if hasattr(self.encoder, 'output_channels'):
            channels_per_stage = self.encoder.output_channels
        else:
            raise AttributeError("El encoder de nnU-Net no expone 'output_channels'.")

        for out_channels in channels_per_stage:
            self.feature_reducers.append(
                nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False)
            )

    def forward(self, x):
        x_ct = x[:, 0:1]
        x_pet = x[:, 1:2]

        # Extracción de características independientes
        skips_ct = self.encoder(x_ct)
        skips_pet = self.encoder(x_pet)

        # Fusión intermedia por proyección de canales
        skips_joint = []
        for i, (s_ct, s_pet) in enumerate(zip(skips_ct, skips_pet)):
            concat_s = torch.cat([s_ct, s_pet], dim=1)
            reduced_s = self.feature_reducers[i](concat_s)
            skips_joint.append(reduced_s)

        # Reconstrucción espacial en paralelo
        out_ct = self.decoder_ct(skips_ct)
        out_pet = self.decoder_pet(skips_pet)
        out_joint = self.decoder_joint(skips_joint)

        if self.training:
            # Solo devolvemos las predicciones espaciales
            return out_joint, out_pet, out_ct
        else:
            # En inferencia evaluamos la predicción conjunta
            return out_joint


# =====================================================================
# --- 2. ENTRENADOR DE NNU-NET ---
# =====================================================================
class nnUNetTr3_PreTr(nnUNetTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        if not self.is_ddp or self.local_rank == 0:
            tb_dir = os.path.join(self.output_folder, "tensorboard_logs")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            
            self.epoch_losses = {
                'Total': [], 
                'Joint': [], 
                'PET': [], 
                'CT': []
            }
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        base_unet = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            1, 
            num_output_channels,
            enable_deep_supervision
        )
        
        network = PETCTSharedEncoderNet(base_unet, num_output_channels, enable_deep_supervision)
        return network
        
    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
            
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        mod.decoder_pet.deep_supervision = enabled
        mod.decoder_ct.deep_supervision = enabled
        mod.decoder_joint.deep_supervision = enabled
        mod.do_ds = enabled

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        
        # Limpieza de carga de datos adaptada a Deep Supervision
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Desempaquetado limpio sin el bottleneck latente
            joint_preds, pet_preds, ct_preds = self.network(data)
            
            l_joint = self.loss(joint_preds, target)
            l_pet = self.loss(pet_preds, target)
            l_ct = self.loss(ct_preds, target)
            
            # Ponderación pura: Joint(1.0) y Asistentes Modalidad (0.33)
            l_total = l_joint + (0.3 * l_pet) + (0.3 * l_ct)

        # Optimización
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
            
        # Registro escalar
        if not self.is_ddp or self.local_rank == 0:
            self.epoch_losses['Total'].append(l_total.item())
            self.epoch_losses['Joint'].append(l_joint.item())
            self.epoch_losses['PET'].append(l_pet.item())
            self.epoch_losses['CT'].append(l_ct.item())
            
        return {'loss': l_joint.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: list):
        super().on_train_epoch_end(train_outputs)
        
        if not self.is_ddp or self.local_rank == 0:
            if len(self.epoch_losses['Total']) > 0:
                mean_total = sum(self.epoch_losses['Total']) / len(self.epoch_losses['Total'])
                mean_joint = sum(self.epoch_losses['Joint']) / len(self.epoch_losses['Joint'])
                mean_pet = sum(self.epoch_losses['PET']) / len(self.epoch_losses['PET'])
                mean_ct = sum(self.epoch_losses['CT']) / len(self.epoch_losses['CT'])
                
                self.tb_writer.add_scalar('Losses_Epoch/1_Total', mean_total, self.current_epoch)
                self.tb_writer.add_scalar('Losses_Epoch/2_Joint_Main', mean_joint, self.current_epoch)
                self.tb_writer.add_scalar('Losses_Epoch/3_PET_Aux', mean_pet, self.current_epoch)
                self.tb_writer.add_scalar('Losses_Epoch/4_CT_Aux', mean_ct, self.current_epoch)
                
            for key in self.epoch_losses.keys():
                self.epoch_losses[key].clear()