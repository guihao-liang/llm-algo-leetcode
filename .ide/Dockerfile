FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN conda env create -f environment.yml \
    && conda clean -afy

SHELL ["bash", "-lc"]

RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate llm_algo \
    && pip install -r requirements/gpu.txt

ENV PATH="/opt/conda/envs/llm_algo/bin:${PATH}"

