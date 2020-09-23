# Docker

```bash
docker build .
docker tag {image_id} gcr.io/awesome-dialect-184820/proteinsolver:v0.1.1
docker push
```

```bash
export PORT=9999
export NOTEBOOK_PATH=proteinsolver/notebooks/30_design_dashboard.ipynb
docker run -p ${PORT}:${PORT} --env=PORT=${PORT} --env=NOTEBOOK_PATH=${NOTEBOOK_PATH} 7b4b6b3be998
```
