FROM nvidia/cuda:10.1-runtime-ubuntu18.04

RUN apt-get update \
    && apt-get install -yq \
        curl \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /env \
    && curl -sS https://storage.googleapis.com/conda-envs/defaults/defaults-v22.tar.gz \
        | tar -xz -C /env \
    && /bin/bash -c "source /env/bin/activate && conda-unpack && source deactivate"

ENV USER_NAME=lucky \
    USER_ID=8002 \
    GROUP_NAME=luckyg \
    GROUP_ID=7300

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} \
    && useradd --no-log-init --create-home -g ${GROUP_NAME} -u ${USER_ID} ${USER_NAME}

WORKDIR /home/${USER_NAME}

USER ${USER_NAME}:${GROUP_NAME}

RUN mkdir -p proteinsolver/notebooks/sudoku_train/c8de7e56 \
    && curl -sS https://storage.googleapis.com/proteinsolver/v0.1/notebooks/sudoku_train/c8de7e56/e1-s400000-d2400000-amv07269.state \
        -o proteinsolver/notebooks/sudoku_train/c8de7e56/e1-s400000-d2400000-amv07269.state \
    && curl -sS https://storage.googleapis.com/proteinsolver/v0.1/notebooks/sudoku_train/c8de7e56/model.py \
        -o proteinsolver/notebooks/sudoku_train/c8de7e56/model.py \
    && mkdir -p proteinsolver/notebooks/protein_train/191f05de \
    && curl -sS https://storage.googleapis.com/proteinsolver/v0.1/notebooks/protein_train/191f05de/e53-s1952148-d93703104.state \
        -o proteinsolver/notebooks/protein_train/191f05de/e53-s1952148-d93703104.state \
    && curl -sS https://storage.googleapis.com/proteinsolver/v0.1/notebooks/protein_train/191f05de/model.py \
        -o proteinsolver/notebooks/protein_train/191f05de/model.py

RUN curl -sS https://gitlab.com/ostrokach/proteinsolver/-/archive/268bbef2a575df90b9c2e006daae003da72de015/proteinsolver-268bbef2a575df90b9c2e006daae003da72de015.tar.gz -O \
    && tar -xzf proteinsolver-268bbef2a575df90b9c2e006daae003da72de015.tar.gz \
    && rm proteinsolver-268bbef2a575df90b9c2e006daae003da72de015.tar.gz \
    && mv proteinsolver-268bbef2a575df90b9c2e006daae003da72de015 proteinsolver

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]

CMD voila --no-browser --ExecutePreprocessor.timeout=3600 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='{"hide"}' --Voila.ip=0.0.0.0 --port=${PORT} ${NOTEBOOK_PATH}
