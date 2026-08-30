FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 1. Usuario y estructura de carpetas
RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

WORKDIR /opt/algorithm

# 2. Dependencias
COPY requirements.txt /opt/algorithm/
RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt

# 3. Usuario algorithm
USER algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# 4. Variables de entorno de nnUNetv2 (nnUNet_results es la que se usa; las otras
#    quedan por compatibilidad, el codigo escribe intermedios en /tmp).
ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"

# 5. Codigo, clasificador y pesos
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm tracer_classifier.py /opt/algorithm/
COPY --chown=algorithm:algorithm rf_tracer_classifier_ds003.joblib /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results
# custom_trainers debe contener: nnUNetTr4_Interactive1.py, nnUNetTr4_Interactive2.py
# y simulate_scribbles.py (lo importan los trainers).
COPY --chown=algorithm:algorithm custom_trainers /opt/algorithm/custom_trainers

ENTRYPOINT ["python", "-m", "process"]