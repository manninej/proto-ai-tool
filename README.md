# proto-codegen

Initial scaffolding for a future codegen/analysis tool. This package currently supports discovering available models from an OpenAI-compatible API.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Set environment variables as needed:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com"  # default
export PROTO_CODEGEN_CANDIDATE_MODELS="gpt-oss-120b,another-model"
```

## CLI usage

```bash
proto-codegen version
proto-codegen models
```

### Options

```bash
proto-codegen models \
  --base-url https://api.openai.com \
  --api-key $OPENAI_API_KEY \
  --timeout 30 \
  --prefer-endpoint auto \
  --candidate gpt-oss-120b \
  --json
```

## Notes

* Model discovery first tries `GET /v1/models` and falls back to probing candidates via `POST /v1/chat/completions` when blocked or unavailable.
* All HTTP calls use `httpx` with small, bounded retries for 429 and 5xx responses.
