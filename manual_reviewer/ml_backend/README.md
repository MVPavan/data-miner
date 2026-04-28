# manual_reviewer ML backend

Pure protocol adapter between Label Studio and the SAM 3.1 LitServe server.
No model weights live here — every smart-tool action turns into an HTTP call
to SAM 3.1 (port 3014), and task-open seeding is a pure sqlite SELECT against
`pipeline.db`.

## Routes

| LS draft type            | Route             | Backend call                                  |
|--------------------------|-------------------|-----------------------------------------------|
| `keypointlabels`         | `smart_click`     | `Sam3OneHttpClient.click_mask` (point→mask)   |
| `textarea`               | `smart_text`      | `Sam3OneHttpClient.text_detect` (text→detect) |
| no context (task open)   | `batch_proposals` | sqlite SELECT on `proposals` table            |

`batch_proposals` runs even when SAM 3.1 is unreachable, so reviewers always
see the cached pipeline output as soon as a task opens.

## Environment

| Var                     | Required | Default                       | Purpose                                  |
|-------------------------|----------|-------------------------------|------------------------------------------|
| `AAV4_PIPELINE_DB`      | yes      | —                             | Path to the pipeline.db (read-only)      |
| `SAM3_1_URL`            | no       | `http://localhost:3014/predict` | SAM 3.1 LitServe endpoint              |
| `SAM3_1_TIMEOUT`        | no       | `60`                          | HTTP timeout in seconds                  |
| `ML_MODEL_VERSION`      | no       | `manual_reviewer_v1`          | Returned to LS as predictions[].version  |
| `LABEL_STUDIO_ML_PORT`  | no       | `9090`                        | Port the Flask harness binds             |
| `LABEL_STUDIO_ML_HOST`  | no       | `0.0.0.0`                     | Bind host                                |

## Running

In docker-compose.review.yml the `ml_backend` service is configured already.
For a local dev loop:

```sh
export AAV4_PIPELINE_DB=/abs/path/to/pipeline.db
export SAM3_1_URL=http://localhost:3014/predict
.venv/bin/python -m manual_reviewer.ml_backend.server
```

Then point the Label Studio project's "Machine learning" settings at
`http://<host>:9090`.

## Failure modes

* SAM 3.1 down → smart routes log a warning and return an empty result; LS
  shows the reviewer's draft alone (no seeded box). `batch_proposals` is
  unaffected — it only touches sqlite.
* `pipeline.db` missing or unreadable → `batch_proposals` returns `[]`. LS
  still shows the `predictions` seeded by `build_tasks.py` from finalize.
* The backend never imports torch / sam3 / transformers, so a CPU-only
  container is sufficient.
