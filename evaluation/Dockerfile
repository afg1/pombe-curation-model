FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
RUN mkdir /workdir
WORKDIR /workdir

RUN pip install "numpy<2" scipy
RUN pip install transformers evaluate datasets
RUN pip install runpod huggingface_hub
RUN pip install scikit-learn
COPY run.sh .
COPY run_evaluation.py .

CMD ["bash", "run.sh"]