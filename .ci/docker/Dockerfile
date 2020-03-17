FROM nvidia/cuda:10.1-runtime-ubuntu18.04

ARG PS_VERSION=v0.1.13

RUN apt-get -qq update \
    && apt-get -y -qq install \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /env \
    && curl -sS https://conda-envs.proteinsolver.org/defaults/defaults-v27.tar.gz | tar -xz -C /env

RUN /bin/bash -c '\
    source /env/bin/activate \
    && conda-unpack \
    && source deactivate'

ENV USER_NAME=jovyan \
    USER_ID=1000 \
    GROUP_NAME=jovyan \
    GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} \
    && useradd --no-log-init --create-home -g ${GROUP_NAME} -u ${USER_ID} ${USER_NAME}

WORKDIR /home/${USER_NAME}

USER ${USER_NAME}:${GROUP_NAME}

RUN JUPYTER_DATA_DIR=$(/env/bin/python -m jupyter --data-dir) \
    && mkdir -p ${JUPYTER_DATA_DIR}/voila/templates/mytemplate \
    && cp -r /env/share/jupyter/voila/templates/default/nbconvert_templates ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/ \
    && cp -r /env/share/jupyter/voila/templates/default/templates ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/ \
    && mkdir ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/static \
    && echo "hello world" >> ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/static/hello.txt

RUN curl -sS https://gitlab.com/ostrokach/proteinsolver/-/archive/${PS_VERSION}/proteinsolver-${PS_VERSION}.tar.gz -O \
    && tar -xzf proteinsolver-${PS_VERSION}.tar.gz \
    && rm proteinsolver-${PS_VERSION}.tar.gz \
    && mv proteinsolver-${PS_VERSION} proteinsolver \
    && cd proteinsolver \
    && wget -q -r -nH --cut-dirs 1 --reject "index.html*" "http://models.proteinsolver.org/v0.1/"

RUN /bin/bash -c '\
    source /env/bin/activate \
    && python -m pip install -q --no-use-pep517 -e proteinsolver \
    && source deactivate'

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]

CMD voila --no-browser --ExecutePreprocessor.timeout=3600 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='{"hide"}' --template=mytemplate --Voila.ip=0.0.0.0 --port=${PORT} ${NOTEBOOK_PATH}
