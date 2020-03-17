FROM registry.gitlab.com/ostrokach/proteinsolver:v0.1.13

RUN id \
    && ls -al

RUN mv proteinsolver proteinsolver-bak \
    && cd proteinsolver-bak \
    && cp . .. \
    && cd .. \
    && rm -r proteinsolver-bak
