# SAGA Code

Initial scaffolding for SAGA Code. This package currently supports discovering available models from an OpenAI-compatible API.

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
export SAGA_CODE_CANDIDATE_MODELS="gpt-oss-120b,another-model"
```

On first run, `saga chat` will prompt for a server URL, access token, optional PEM bundle, and a default model. These
values are stored in YAML at:

* `$XDG_CONFIG_HOME/saga-code/config.yaml` (if `XDG_CONFIG_HOME` is set)
* `~/.config/saga-code/config.yaml`

## CLI usage

```bash
saga version
saga models
saga chat --model gpt-oss-120b
```

### Options

```bash
saga models \
  --base-url https://api.openai.com \
  --api-key $OPENAI_API_KEY \
  --timeout 30 \
  --prefer-endpoint auto \
  --candidate gpt-oss-120b \
  --json
```

### Chat usage

```bash
saga chat --model gpt-oss-120b
```

Use `/quit` or `/exit` to leave the chat session.

During chat, you can update configuration with:

* `/server [url] [pem]` - change server URL and optional PEM bundle path
* `/token [token]` - change access token
* `/model [model-id]` - change model (omit the model ID to list and select)

## Notes

* Model discovery first tries `GET /v1/models` and falls back to probing candidates via `POST /v1/chat/completions` when blocked or unavailable.
* All HTTP calls use `httpx` with small, bounded retries for 429 and 5xx responses.
