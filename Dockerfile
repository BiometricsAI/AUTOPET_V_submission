FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 1. Crear usuario y estructura de carpetas (como administrador)
RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs && \
    mkdir -p /opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

WORKDIR /opt/algorithm

# 2. Instalar dependencias a nivel global (como administrador)
COPY requirements.txt /opt/algorithm/
RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt

# 3. Cambiar al usuario algorithm
USER algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# 4. Variables de entorno OBLIGATORIAS para nnUNetv2
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

# 5. Copiar código y pesos
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm rf_tracer_classifier_ds003.joblib /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results
COPY --chown=algorithm:algorithm custom_trainers /opt/algorithm/custom_trainers

ENTRYPOINT ["python", "-m", "process"]