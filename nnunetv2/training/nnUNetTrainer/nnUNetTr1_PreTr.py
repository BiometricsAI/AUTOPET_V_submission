"""
nnUNetTr1_PreTr: trainer de fase-1 (pretraining).

Alias con nombre propio del nnUNetTrainer. OJO: en este repo el nnUNetTrainer
base YA esta modificado (usa MetaLogger, extrae la flag 'continue_training' del
dict de plans y vuelca los hiperparametros al logger). Por tanto esta subclase
NO debe repetir nada de eso: toda la logica esta en la clase base.

Solo existe para tener un nombre de trainer distinto, de modo que:
  - el output folder sea  .../nnUNetTr1_PreTr__<plans>__<config>
  - el checkpoint se etiquete con trainer_name='nnUNetTr1_PreTr' y la inferencia
    reinstancie esta misma clase.

La arquitectura la lee de los plans (build_network_architecture heredado), asi
que Plain vs ResEnc se controla con el flag -p al entrenar.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTr1_PreTr(nnUNetTrainer):
    pass