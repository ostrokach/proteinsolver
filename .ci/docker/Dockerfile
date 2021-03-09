FROM nvidia/cuda:10.1-runtime-ubuntu18.04

ARG PS_VERSION=v0.1.25

RUN apt-get -qq update \
    && apt-get -y -qq install \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /env \
    && curl -sS https://conda-envs.proteinsolver.org/defaults/defaults-v28.tar.gz | tar -xz -C /env

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
    && google_analytics_tag='<script async src="https://www.googletagmanager.com/gtag/js?id=UA-72981430-5"></script>\n<script>window.dataLayer = window.dataLayer || []; function gtag(){dataLayer.push(arguments);} gtag("js", new Date()); gtag("config", "UA-72981430-5");</script>\n{%- endblock html_head_css -%}' \
    && sed -i "s@{%- endblock html_head_css -%}@${google_analytics_tag}@" ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/nbconvert_templates/voila.tpl \
    && mkdir ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/static \
    && echo "hello world" >> ${JUPYTER_DATA_DIR}/voila/templates/mytemplate/static/hello.txt

RUN curl -sS https://gitlab.com/ostrokach/proteinsolver/-/archive/${PS_VERSION}/proteinsolver-${PS_VERSION}.tar.gz -O \
    && tar -xzf proteinsolver-${PS_VERSION}.tar.gz \
    && rm proteinsolver-${PS_VERSION}.tar.gz \
    && mv proteinsolver-${PS_VERSION} proteinsolver \
    && cd proteinsolver \
    && wget -q -r -nH --cut-dirs 1 --reject "index.html*" "http://models.proteinsolver.org/v0.1/"

COPY favicon.ico 1n5uA03-distance-matrix.txt proteinsolver/notebooks/

RUN /bin/bash -c '\
    source /env/bin/activate \
    && python -m pip install -q --no-use-pep517 -e proteinsolver \
    && pip install --upgrade sentry-sdk \
    && source deactivate'

COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]

CMD voila --no-browser --MappingKernelManager.cull_interval=60 --MappingKernelManager.cull_idle_timeout=600 --ExecutePreprocessor.timeout=3600 --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='{"hide"}' --VoilaConfiguration.file_whitelist="['favicon.ico', '1n5uA03-distance-matrix.txt']" --template=mytemplate --Voila.ip=0.0.0.0 --port=${PORT} ${NOTEBOOK_PATH}
