FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
RUN mkdir /workdir
WORKDIR /workdir

COPY requirements.txt .

RUN pip install polars transformers datasets evaluate "numpy<2" scikit-learn wandb huggingface_hub

COPY fetch_data.sh .
COPY process_data.py .
COPY five_fold.py .
COPY run.sh .

RUN pip install "accelerate>=0.21.0"

ENV WANDB_PROJECT "pombe_curation_model"
ENV WANDB_LOG_MODEL "checkpoint"

CMD ["bash", "run.sh"]