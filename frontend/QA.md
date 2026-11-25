# RallyMirror verification

Executed on 25 November 2026 with Node 24.14.0, Python 3.13.7, installed
Microsoft Edge and Docker Desktop's Linux engine on Windows.

## Executed checks

| Check | Observed result |
| --- | --- |
| TypeScript and Vite production build | Passed |
| Vitest | 37/37 passed |
| FastAPI/store/worker contract tests | 14/14 passed |
| Offline model-fetch tests | 6/6 passed |
| Edge browser suite | 40/40 passed in 1.8 minutes |
| API and CPU worker Docker images | Both built successfully |
| Selected model preflight | All six SHA-256 hashes verified |
| Real container court upload | Succeeded; court calibration available |
| Real container tracking upload | Succeeded; 176 player observations |
| Real container combined upload | Succeeded; all six modules available |
| Production browser with the real private job | Video played, report loaded and six module counts displayed |
| Production rendering/performance checks | Passed |

The browser suite verifies five routes at 1440 × 1000 and 390 × 844, plus
320-pixel layout checks. It covers native camera acts, text reveal completion,
gallery/list links and keyboard use, reduced motion, WebGL fallback/context
loss, input validation, seven modes, upload and polling, private authorization,
signed-link refresh, history persistence, warnings, seven real example videos
and source/combined comparison. There were no uncaught page errors.

API lifecycle cases use explicit fake executors so their results cannot be
mistaken for model evaluation. They verify each mode through creation, upload,
queue, authenticated execution and signed artifact access; legacy SQLite
migration; invalid modes; static-serving boundaries; and cloud-worker mode
claims. On Windows these checks need an unrestricted local socket for the
asyncio test event loop. A sandbox blocked socket-pair initialization before
application startup; the outside-sandbox run passed. The only pytest warning
was Starlette's test-client deprecation notice for httpx.

## Actual inference evidence

One four-second, 120-frame, 1280 × 720, 30 fps raw rally was processed on CPU
using the six selected checkpoints. No manual ground-truth labels were used
to draw the predictions. The seven public example views were rendered from
that same run's saved artifacts.

| Module | Windows example run | Linux container acceptance run |
| --- | ---: | ---: |
| Court calibration | Accepted | Accepted |
| Player observations | 173 | 176 |
| Body-pose observations | 173 | 176 |
| Displayable racket centres | 173 | 176 |
| Detected shuttle frames | 119 | 119 |
| Player/frame phase predictions | 77 | 78 |

The Windows run took 142.5 seconds. The fresh combined upload through the
Linux API/worker completed in about 125 seconds, including queue/poll delay.
These measurements describe this machine and clip; they are not real-time
throughput promises. Differences across platforms are retained in the reports.
Counts measure available predictions, not correctness or model accuracy.

The first Linux combined acceptance attempt identified a missing MediaPipe
native GLES library. The worker image now installs `libgles2`; the rebuilt
image completed the full run. Both images include the final source snapshot.
The worker uses the tested CPU pair Torch 2.14.1 / torchvision 0.29.2,
MediaPipe 1.1.0 and Ultralytics 8.4.174.

The live acceptance path exercised MP4 creation/upload, persisted mode, worker
queue claims, real inference, browser-compatible H.264 output, private report
access and HTTP 206 Range playback. The finished production browser also
loaded and played that real private result. Static example media hashes and
their reports are checked by the browser suite against
`public/media/demo-provenance.json`.

The example is demonstration footage, not a held-out benchmark. No overall
accuracy, independent-court generalization or physical shuttle speed is claimed.
The five phase classes are ready, movement, stroke, recovery and idle. Warmup
and missing detections remain unknown. Court mapping describes the floor plane.

## Rendering observations

Measured against the production container build in installed Edge:

| Observation | Result |
| --- | --- |
| Visible court over three seconds | 77 render passes; 41.4 ms median interval |
| Fully offscreen over 1.5 seconds | Zero continuing render passes |
| Returned to view over one second | 27 render passes |
| Motion disabled over 1.5 seconds | Zero continuing render passes |
| Fresh upload route | No CourtScene or Three.js renderer chunk requested |
| Uncaught page errors | None |

The observed pixel ratio was about 1 on this display; source caps it at 1.5
(1.25 for narrow scenes). Rendering has a 30 fps ceiling. This is not a
universal zero-lag guarantee. Mobile checks use browser emulation, not physical
phones. Production desktop and mobile captures were reviewed for clipping,
legibility, result warnings and module counts.

## Reproduce

From `frontend`:

```powershell
npm run build
npm test
python -m unittest discover -s deployment -p test_fetch_models.py
# Start npm run dev in another terminal:
npm run test:e2e
```

From `frontend/backend`, using the backend environment:

```powershell
python -m pytest tests -q
```

For another browser-test server, set `E2E_BASE_URL` before running the suite.
For rendering measurements, set `PERF_BASE_URL` to a production server before
`npm run check:performance`; its default is the Vite preview on port 4173.

Temporary logs, private test credentials, snapshots and model artifacts remain
ignored under `.work/`, `.runtime/` and the backend's output directories. No
public cloud/GPU service or domain was deployed. See [DEPLOYMENT.md](DEPLOYMENT.md).
