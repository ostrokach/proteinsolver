# Docker

```bash
docker build .
docker tag {image_id} gcr.io/awesome-dialect-184820/proteinsolver:v0.1.1
docker push
```

```bash
PORT=8080
# NOTEBOOK_PATH=proteinsolver/notebooks/30_sudoku_dashboard.ipynb
NOTEBOOK_PATH=proteinsolver/notebooks/30_design_dashboard.ipynb
docker run -d --restart unless-stopped --gpus device=0 --publish ${PORT}:${PORT} \
  registry.gitlab.com/ostrokach/proteinsolver:v0.1.25 \
  voila --no-browser \
  --MappingKernelManager.cull_interval=60 --MappingKernelManager.cull_idle_timeout=600 \
  --ExecutePreprocessor.timeout=3600 \
  --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='{"hide"}' \
  --VoilaConfiguration.file_whitelist="['favicon.ico', '1n5uA03-distance-matrix.txt']" \
  --template=mytemplate --Voila.ip=0.0.0.0 --port=${PORT} ${NOTEBOOK_PATH}
```
