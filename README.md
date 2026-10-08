# BlackSmoke LLM

A FastAPI service that runs LLM analysis tasks on app-store reviews loaded from CSV files: summaries, topic discovery, per-review topic tagging, bug and feature-request extraction, action and marketing recommendations, and sub-topic extraction.

Each task has its own model route in YAML: a primary provider/model plus ordered fallbacks. OpenAI and Gemini are called through their native SDKs.

## Quick start

```bash
uv sync
cp .env.example .env   # then fill in the keys
uv run uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the typed API, or call `GET /v1/tasks` to list tasks and their routes.

```bash
curl -X POST localhost:8000/v1/tasks/find_bugs_and_features \
  -H 'content-type: application/json' \
  -d '{"file_path": "reviews_2025_02_22_210638_com.gardrops.csv"}'
```

Run the tests (no network calls) with `uv run pytest`, and lint with `uv run ruff check .`.

The Docker image contains neither data nor secrets. Mount the CSVs and pass the environment at runtime:

```bash
docker build -t blacksmoke-llm .
docker run -p 8000:8000 --env-file .env -v "$PWD/data:/app/data" blacksmoke-llm
```

## Environment

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_ORG_ID` | OpenAI credentials |
| `GOOGLE_API_KEY` | Gemini credentials |
| `BLACKSMOKE_CONFIG_DIR` | Optional; defaults to `./config` |

A provider without a key is skipped at call time and the task falls back to its next target.

## Architecture

```text
blacksmoke/
  api/        FastAPI app, generated per-task routes, error mapping (HTTP only)
  services/   TaskService (orchestration), AggregateRunner, PerReviewRunner, response envelopes
  tasks/      Task definitions, registry, schemas, prompt rendering
  llm/        provider port, OpenAI/Gemini adapters, factory, retry/repair/fallback gateway, pricing
  data/       CSV repository, review filtering, result writer
  core/       settings (.env), YAML config loading, composition root (container)
config/       app.yaml, tasks.yaml, models.yaml, prompts/<task>/{system,user}.md.j2
```

Dependencies point inward: the API calls services, services use tasks, the LLM layer and the data layer, and tasks never import an SDK or FastAPI.

A request flows like this:

1. The generated route validates the task's request model and calls `TaskService.run`.
2. The service looks up the task in the `TaskRegistry` and resolves the model route from `tasks.yaml`, applying the request's optional `model_override` and `allow_fallback`.
3. It loads the CSV through `CsvReviewRepository`, which only allows paths inside `data_dir`. The task selects its reviews, then the review filter applies.
4. The matching runner renders the Jinja prompts and calls `LLMGateway`:
   - Aggregate tasks send all reviews in one call.
   - Per-review tasks make one call per review with bounded concurrency.
5. The gateway tries each target in the route. Each target is wrapped as `StructuredRepair(RetryingProvider(RecordingProvider(adapter)))`.

Failure handling in the gateway:

| Failure | Behaviour |
|---|---|
| Timeout, connection error, 5xx, rate limit | Retried on the same target with jittered backoff, then fall back |
| Output fails schema validation | Re-asked with the validation error, then fall back |
| Output truncated | `max_output_tokens` raised (up to a cap), then fall back |
| Auth error, model not found, provider not configured | Fall back immediately |
| Bad request | Fail fast (HTTP 502 with the attempt trail) |

Every physical call is recorded as an attempt and priced against the model that actually ran. Responses report `served_by`, token usage, `cost_usd` and the full `attempts` list.

### Task kinds

- **Aggregate tasks** (`AggregateTask`) return `TaskResult[Output]`.
- **`tag_reviews_with_topics`** (`PerReviewTask`) writes a CSV in the `tagged` format to `output_dir` and returns a `TaggingReport` with the file path. Pass that path as `file_path` to `recommend_actions`, `recommend_marketing_strategies` or `extract_subtopics`.

## Configuration

- **`config/app.yaml`:** the data and output directories, CSV formats (`raw` uses `;`, `tagged` uses `¤` plus topic list columns), the default review filter and the prompt delimiter.
- **`config/tasks.yaml`:** per task, the `route`, generation `params`, `resilience` overrides, `review_filter` overrides and free-form `settings`. Params merge as: defaults, then task params, then route entry params. Use `provider_options` for provider-specific options, e.g. `{thinking_budget: 0}` for Gemini or `{reasoning: {effort: minimal}}` for OpenAI reasoning models.
- **`config/models.yaml`:** the model catalog with USD prices per 1M tokens. Request overrides must name a model listed here.
- **`config/prompts/<task>/`:** `system.md.j2` and `user.md.j2`. Templates use strict undefined variables, so a missing variable fails at startup.

The app validates code, YAML, providers and prompts together on startup and refuses to start if they disagree.

## Adding a task

1. Create `blacksmoke/tasks/<name>.py` with a class extending `AggregateTask` or `PerReviewTask`, decorated with `@register_task`. Set `name`, `summary`, `request_model`, `output_model` and `example_request`. Override `select_reviews`, `prompt_vars`, `postprocess` (aggregate tasks) or `review_row`/`failed_row` (per-review tasks) as needed.
2. Add the request and output models to `blacksmoke/tasks/schemas.py`. Output fields must be required, with no defaults.
3. Add `config/prompts/<name>/system.md.j2` and `user.md.j2`.
4. Add a `tasks.yaml` entry with a route.

The route `POST /v1/tasks/<name>` is generated automatically.

## Adding a provider

1. Implement the `LLMProvider` protocol (`blacksmoke/llm/provider.py`) in `blacksmoke/llm/providers/`. Translate SDK errors into `blacksmoke.llm.errors` types, raise `OutputTruncatedError` and `OutputValidationError` for bad output, and do not retry inside the adapter.
2. Register a builder in `default_provider_builders` (`blacksmoke/core/container.py`).
3. Add its models and prices to `models.yaml`, then reference it in task routes.
