# Deployment

- [Deploying to local machines](#deploying-to-local-machines)
  - [Sudoku web server](#sudoku-web-server)
  - [Protein design web server](#protein-design-web-server)
- [Deploying to Kubernetes using Knative](#deploying-to-kubernetes-using-knative)
  - [Sudoku web server](#sudoku-web-server-1)
  - [Protein design web server](#protein-design-web-server-1)

## Deploying to local machines

### Sudoku web server

```bash
docker run -d --restart unless-stopped -p 8080:8080 \
  --env PORT=8080 \
  --env NOTEBOOK_PATH=proteinsolver/notebooks/30_sudoku_dashboard.ipynb \
  registry.gitlab.com/ostrokach/proteinsolver:v0.1.15
```

### Protein design web server

```bash
docker run -d --restart unless-stopped -p 8080:8080 \
  --env PORT=8080 \
  --env NOTEBOOK_PATH=proteinsolver/notebooks/30_design_dashboard.ipynb \
  --gpus '"device=0"' \
  registry.gitlab.com/ostrokach/proteinsolver:v0.1.15
```

## Deploying to Kubernetes using Knative

### Sudoku web server

Create a Sudoku service configuration file, as given below.

```yaml
# sudoku-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: sudoku
  namespace: default
spec:
  template:
    metadata:
      name: sudoku-v1
      annotations:
        # Knative concurrency-based autoscaling (default).
        autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
        autoscaling.knative.dev/metric: concurrency
        # Disable scale to zero with a minScale of 1.
        autoscaling.knative.dev/minScale: "1"
        # Limit scaling to 10 pods.
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
        - name: user-container
          image: registry.gitlab.com/ostrokach/proteinsolver:v0.1.15
          ports:
            - containerPort: 8080
          env:
            - name: NOTEBOOK_PATH
              value: proteinsolver/notebooks/30_sudoku_dashboard.ipynb
          resources:
            limits:
              cpu: "1.8"
              memory: 12Gi
      timeoutSeconds: 600
      containerConcurrency: 12
```

Apply the Sudoku service configuration file.

```bash
kubectl apply -f sudoku-service.yaml
```

### Protein design web server

Create a protein design service configuration file, as given below.

```yaml
# design-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: design
  namespace: default
spec:
  template:
    metadata:
      name: design-v1
      annotations:
        # Knative concurrency-based autoscaling (default).
        autoscaling.knative.dev/class: kpa.autoscaling.knative.dev
        autoscaling.knative.dev/metric: concurrency
        # Disable scale to zero with a minScale of 1.
        autoscaling.knative.dev/minScale: "1"
        # Limit scaling to 10 pods.
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
        - name: user-container
          image: registry.gitlab.com/ostrokach/proteinsolver:v0.1.15
          ports:
            - containerPort: 8080
          env:
            - name: NOTEBOOK_PATH
              value: proteinsolver/notebooks/30_design_dashboard.ipynb
          resources:
            limits:
              cpu: "7"
              memory: 24Gi
              nvidia.com/gpu: "1"
      timeoutSeconds: 600
      containerConcurrency: 8
```

Apply the protein design service configuration file.

```bash
kubectl apply -f design-service.yaml
```
