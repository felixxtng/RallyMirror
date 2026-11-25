# Deploy RallyMirror Studio

All frontend source, the API/worker snapshot, selected model paths and deployment
files live inside `frontend/`. The Docker build context is this folder.

The API serves the built React application and `/api/*` from one origin. A
separate CPU worker reads the same durable SQLite queue and writes each job's
video/report into a shared volume. Uploaded analysis can select `combined`,
`court`, `tracking`, `pose`, `racket`, `shuttle` or `phases`. The API records that
selection and passes it to the worker.

Both Linux images were built and verified locally with fresh court, tracking
and combined uploads, including all six models and private video playback.
See [QA.md](QA.md) for the execution record. Public cloud services and a domain
have not been provisioned. Browser rendering and model inference time are
separate measurements.

## Prepare the selected models

Python 3.13 is the target runtime. From the `frontend` directory:

```powershell
python scripts/fetch-models.py
python scripts/fetch-models.py --check
```

The helper explicitly downloads missing selected files from RallyMirror commit
`017a366aed9ec4091881b678412e14e2eca518f8` and checks all six SHA-256 hashes.
Existing matching files need no download. A different checkpoint stops the
operation and stays unchanged. The helper installs no Python packages and runs
no training or inference. It has no effect until you run it.

Models are in `backend/models/` and excluded from Git by default. They are
included in the worker's Docker build context, so prepare them before building.
The worker verifies them again during build and startup. Model selection is
recorded in `backend/configs/deployment_models.json`; changing a checksum to
bypass a mismatch is not a deployment fix.

## Run with Docker Compose

Install and start Docker Desktop with its Linux engine enabled. From `frontend`:

```powershell
Copy-Item deployment/local.env.example deployment/local.env
docker compose --env-file deployment/local.env -f deployment/compose.yaml config --quiet
docker compose --env-file deployment/local.env -f deployment/compose.yaml up --build -d
```

Open **http://127.0.0.1:8000/**. The same address serves the website and API.
Use `RALLYMIRROR_PORT` in `deployment/local.env` if port 8000 is already occupied.
The default compose binding is loopback, so it is reachable from this computer.

The first build fetches npm/Python dependencies and the two base container
images. The worker installs the verified Torch `2.14.1` / torchvision `0.29.2`
CPU pair from the official PyTorch wheel index. The API image does not install
Torch or load model weights. The worker includes FFmpeg/ffprobe and the native
GL/EGL/GLES libraries required by MediaPipe, and runs as the same unprivileged
UID as the API.

Set `APP_ACCESS_TOKEN` in the local env file if you want the upload form to
require a private team key. It is a server setting; never put it in a
`VITE_*` variable. Do not commit `deployment/local.env`.

Useful checks:

```powershell
docker compose --env-file deployment/local.env -f deployment/compose.yaml ps
docker compose --env-file deployment/local.env -f deployment/compose.yaml logs --tail 80 api worker
docker compose --env-file deployment/local.env -f deployment/compose.yaml exec worker python -m src.deployment.preflight
```

The API health check is `/healthz`. Test a short raw rally using court-only mode
first, then combined analysis. Confirm that the job reaches `succeeded`, the
video plays and seeks, the report names the selected mode and any quality
warnings, and the model checksums match the expected selection. A saved example
video or mocked API test does not establish fresh model inference.

The CPU worker processes one video at a time. The default limit is 60 seconds
and 200 MiB per upload, with a 30-minute processing timeout. CPU analysis may
take substantially longer than the clip duration. Change these limits and
`WORKER_CPU_THREADS` in the local env file only after checking memory and runtime
on the deployment computer. Adding API processes does not accelerate inference.

Stop the services while preserving jobs:

```powershell
docker compose --env-file deployment/local.env -f deployment/compose.yaml down
```

The named `jobs` volume contains the database, raw uploads, result videos and
reports. It survives a normal `down`; adding `--volumes` would remove it. Back it
up before deliberate deletion. If a worker is interrupted, stop all worker
instances before using its recovery command:

```powershell
docker compose --env-file deployment/local.env -f deployment/compose.yaml stop worker
docker compose --env-file deployment/local.env -f deployment/compose.yaml run --rm worker python -m src.deployment.worker --recover --once
docker compose --env-file deployment/local.env -f deployment/compose.yaml start worker
```

Recovery marks previously running jobs failed rather than treating partial
files as successful. Resubmit a failed input after addressing its logged cause.

## Run directly on a development computer

The build and model preparation commands still run from `frontend`:

```powershell
npm ci
npm run build
python scripts/fetch-models.py --check
```

Install backend dependencies in a separate project environment, then run the
API and worker in two terminals whose working directory is `frontend/backend`.
FFmpeg and ffprobe must be on PATH:

```powershell
# Terminal 1, frontend/backend
python -m pip install -r requirements-api.txt
python -m uvicorn src.deployment.api:create_app --factory --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2, frontend/backend, after installing compatible worker dependencies
$env:INFERENCE_DEVICE = 'cpu'
python -m src.deployment.preflight
python -m src.deployment.worker
```

For a clean CPU environment, install the matching Torch pair from
`https://download.pytorch.org/whl/cpu` first, then `requirements-worker.txt`.
The Dockerfile contains the tested installation commands. Do not
replace an existing working GPU environment's Torch installation just to run
the frontend. `FRONTEND_DIST` defaults to `frontend/dist`, while the API and
worker default to `backend/outputs/deployment` for their shared state.

## Publish behind HTTPS

A single Linux host can run the same two services. Put a TLS reverse proxy in
front of the loopback API port, point the domain at that host and configure a
private upload access key. Proxy both `/` and `/api/*` to the same API service;
keep HTTP Range requests for seeking and raw PUT request bodies intact. Align
the proxy's upload limit with `MAX_UPLOAD_BYTES` and allow long job polling.
Do not make the underlying volume or model directory publicly browsable.

The existing optional cloud backend separates a CPU API, PostgreSQL, private
S3-compatible/R2 media storage and a Runpod worker. The placeholder settings in
`deployment/cloud.env.example` describe that contract. This local Compose file
does not provision those services. A GPU worker requires a separately validated
CUDA Torch image and device checks; the CPU Dockerfile does not become a GPU
image by setting `INFERENCE_DEVICE=cuda`.

Before sharing a hosted address, verify a raw video through the complete upload,
queue, inference and result path, authenticate access, configure HTTPS and media
retention, and test failure/restart behaviour. Static hosting of `dist/` alone
supports the example and local comparison, but cannot execute Python models.

## Offline helper checks

```powershell
python -m unittest discover -s deployment -p test_fetch_models.py
```

These tests cover verified reuse, missing files, hash failures, atomic
installation and path containment without contacting GitHub.
