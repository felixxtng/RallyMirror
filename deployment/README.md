Deployment entry points and environment templates are here. See [the full setup and manual checklist](../docs/deployment.md).

- `Dockerfile.api`: CPU web UI/API; Render Blueprint is `../render.yaml`.
- `Dockerfile.worker`: Runpod queue worker with selected models and FFmpeg.
- `local.env.example` / `cloud.env.example`: variable references, not secrets.
- `r2-cors.json`: replace the origin and apply to the private R2 bucket.
