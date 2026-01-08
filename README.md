# SAGA Code

SAGA Code is a developer-facing CLI (``sage``) and Python package (``saga_code``) for interacting with OpenAI-compatible
APIs and running domain-focused prompt bundles such as chat and C++ explanations.

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

On first run, ``sage chat`` will prompt for a server URL, access token, optional PEM bundle, and a default model. These
values are stored in YAML at:

* ``$XDG_CONFIG_HOME/saga-code/config.yaml`` (if ``XDG_CONFIG_HOME`` is set)
* ``~/.config/saga-code/config.yaml``

## CLI usage

```bash
sage version
sage models
sage chat --model gpt-oss-120b
sage explain-cpp src/
sage explain-cpp --json src/foo.cpp
```

### Options

```bash
sage models \
  --base-url https://api.openai.com \
  --api-key $OPENAI_API_KEY \
  --timeout 30 \
  --prefer-endpoint auto \
  --candidate gpt-oss-120b \
  --json
```

### Chat usage

```bash
sage chat --model gpt-oss-120b
```

Use ``/quit`` or ``/exit`` to leave the chat session.

During chat, you can update configuration with:

* ``/server [url] [pem]`` - change server URL and optional PEM bundle path
* ``/token [token]`` - change access token
* ``/model [model-id]`` - change model (omit the model ID to list and select)

### C++ explain usage

```bash
sage explain-cpp src/
sage explain-cpp --json src/foo.cpp
```

The explain command reads up to 20 files (default) and 200000 bytes total by default. Use ``--show-reasoning`` to emit
the model reasoning panel for debugging.

## Prompt customization

Prompt templates live under ``./prompts`` and are layered by name:

```
prompts/
  active_stack.txt
  default/
  fiat/
  tractors/
  cars/
```

Each layer contains bundles such as ``chat`` or ``explain_cpp``:

```
prompts/default/explain_cpp/system.j2
prompts/default/explain_cpp/user.j2
prompts/fiat/explain_cpp/system.append.j2
prompts/cars/shared/constraints.append.j2
```

Select an active layer stack (in order) with:

```bash
sage prompts default,fiat,cars
```

Inspect how a prompt is composed or rendered:

```bash
sage prompts --show-resolved explain_cpp/system
sage prompts --render chat/system
```

### Variables

Each layer may include a ``variables.yaml`` file. Later layers override earlier ones:

```yaml
abbrev:
  DUT: "Device Under Test"
coding_style:
  cpp: "C++20, RAII, ..."
```

Templates access these as top-level variables.

## Notes

* Model discovery first tries ``GET /v1/models`` and falls back to probing candidates via ``POST /v1/chat/completions`` when blocked or unavailable.
* All HTTP calls use ``httpx`` with small, bounded retries for 429 and 5xx responses.
* Prompt templates and the active stack file are versioned in the repo for airgapped workflows.
