FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
RUN mkdir /workdir
WORKDIR /workdir

COPY requirements.txt .

RUN pip install polars transformers datasets evaluate "numpy<2" scikit-learn

COPY fetch_data.sh .
COPY process_data.py .
COPY five_fold.py .
