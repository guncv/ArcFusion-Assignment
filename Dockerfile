FROM mambaorg/micromamba:latest

# Install system dependencies as root
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /src

COPY --chown=$MAMBA_USER:$MAMBA_USER package.yaml /tmp/freeze_env.yaml
RUN micromamba env create --yes --file /tmp/freeze_env.yaml && micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ACTIVATE_ENVIRONMENT=arcfusion-asm

ENV JAVA_HOME=/opt/conda/envs/arcfusion-asm
ENV PATH="$JAVA_HOME/bin:$PATH"

RUN micromamba run -n arcfusion-asm java -version && echo "Java verified successfully"

COPY . .
COPY gunicorn.conf.py .

CMD ["micromamba", "run", "-n", "arcfusion-asm", "gunicorn", "src.main:app", "--config", "gunicorn.conf.py"]