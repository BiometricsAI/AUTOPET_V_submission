FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime


RUN groupadd -r algorithm && \
    useradd -m --no-log-init -r -g algorithm algorithm && \
    mkdir -p /opt/algorithm /input /output /output/images/tumor-lesion-segmentation && \
    mkdir -p /opt/algorithm/nnUNet_preprocessed && \
    chown -R algorithm:algorithm /opt/algorithm /input /output

WORKDIR /opt/algorithm


COPY requirements.txt /opt/algorithm/
RUN python -m pip install -U pip && \
    python -m pip install -r requirements.txt


USER algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"


ENV nnUNet_raw="/opt/algorithm/nnUNet_raw"
ENV nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed"
ENV nnUNet_results="/opt/algorithm/nnUNet_results"


COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm tracer_classifier.py /opt/algorithm/
COPY --chown=algorithm:algorithm rf_tracer_classifier_ds003.joblib /opt/algorithm/
COPY --chown=algorithm:algorithm nnUNet_results /opt/algorithm/nnUNet_results

COPY --chown=algorithm:algorithm custom_trainers /opt/algorithm/custom_trainers

ENTRYPOINT ["python", "-m", "process"]